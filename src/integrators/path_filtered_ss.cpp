#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

#include <mitsuba/render/manifold_ss.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * This is a mostly a standard path tracer, with the exception that some of the
 * caustic light paths are filtered out.
 *
 * More specifically, it produces the same results as 'path_sms_ss' but relies
 * on standard BSDF / emitter sampling strategies to find the castics that
 * are sampled with specular manifold sampling in the other one. It's purpose
 * are fair comparisons of our technique in a setting where noise from other
 * (even more challenging light paths) is not present.
 *
 * It is used for Figures 6, 14, 17 in the paper.
 *
 */
template <typename Float, typename Spectrum>
class SingleScatterFilteredPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium)

    SingleScatterFilteredPathIntegrator(const Properties &props) : Base(props) {
        m_disable_reflected_emitters = props.bool_("disable_reflected_emitters", false);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return { 0.f, 0.f };
        } else {
            RayDifferential3f ray = ray_;
            SurfaceInteraction3f si_prev;
            Float eta = 1.f;
            Spectrum throughput(1.f), result(0.f);

            // ---------------------- First intersection ----------------------

            SurfaceInteraction3f si = scene->ray_intersect(ray);
            Mask valid_ray = si.is_valid();
            EmitterPtr emitter = si.emitter(scene);

            if (emitter) {
                result += emitter->eval(si);
            }

            // ---------------------- Main loop ----------------------

            for (int depth = 1;; ++depth) {

                // ------------------ Possibly terminate path -----------------

                if (!si.is_valid())
                    break;
                si.compute_partials(ray);

                if (depth > m_rr_depth) {
                    Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                    if (sampler->next_1d() > q)
                        break;
                    throughput *= rcp(q);
                }

                if (uint32_t(depth) >= uint32_t(m_max_depth))
                    break;

                bool on_caustic_caster = si.shape->is_caustic_caster_single_scatter() ||
                                         si.shape->is_caustic_caster_multi_scatter();
                Complex<Spectrum> caustic_ior = si.bsdf()->ior(si);
                Float caustic_eta = select(all(eq(0.f, imag(caustic_ior))), hmean(real(caustic_ior)), 1.f);

                bool from_caustic_receiver = si_prev.is_valid() && si_prev.shape->is_caustic_receiver();

                // --------------------- Emitter sampling ---------------------

                BSDFContext ctx;
                BSDFPtr bsdf = si.bsdf(ray);

                /* To account for exactly the same light paths as in the version
                   with SMS we need to be careful whenever we are on a caustic
                   caster currently.
                   In this case, we account for the direct illumination only if
                   we either
                    - scattered off a caustic receiver just before (thus SMS)
                      could have accounted for this path

                   or
                    - the last bounce was the camera, so we might be computing
                      a direct (glossy) reflection of a light source */
                if (has_flag(bsdf->flags(), BSDFFlags::Smooth)) {
                    bool glint_case = (depth == 1 && !m_disable_reflected_emitters);
                    bool valid_connection = on_caustic_caster && (from_caustic_receiver || glint_case);

                    if (!on_caustic_caster || valid_connection) {
                        auto [ds, emitter_weight] = scene->sample_emitter_direction(si, sampler->next_2d(), true);
                        if (ds.pdf != 0.f) {
                            // Query the BSDF for that emitter-sampled direction
                            Vector3f wo = si.to_local(ds.d);
                            Spectrum bsdf_val = bsdf->eval(ctx, si, wo);
                            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                            // Determine density of sampling that same direction using BSDF sampling
                            Float bsdf_pdf = bsdf->pdf(ctx, si, wo);
                            Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                            /* One last check before adding contribution: if this is a caustic path,
                               make sure it's the type of caustic (reflection vs. refraction)
                               we are sampling with SMS. Which of these to choose is also
                               an open question---for now we always try for transmission on dielectrics
                               even though reflection cases could also happen. */
                            if (valid_connection && depth > 1) {
                                Float cos_theta_x = Frame3f::cos_theta(wo),
                                      cos_theta_y = Frame3f::cos_theta(si.wi);
                                bool refraction = cos_theta_x * cos_theta_y < 0.f;
                                bool reflection = !refraction;
                                if ((caustic_eta == 1.f && !reflection) ||
                                    (caustic_eta != 1.f && !refraction)) {
                                    mis = 0.f;
                                }
                            }

                            result += mis * throughput * bsdf_val * emitter_weight;
                        }
                    }
                }

                // ----------------------- BSDF sampling ----------------------

                // Sample BSDF * cos(theta)
                auto [bs, bsdf_weight] = bsdf->sample(ctx, si, sampler->next_1d(),
                                                      sampler->next_2d());
                bsdf_weight = si.to_world_mueller(bsdf_weight, -bs.wo, si.wi);

                throughput = throughput * bsdf_weight;
                eta *= bs.eta;

                if (all(eq(throughput, 0.f)))
                    break;

                // Intersect the BSDF ray against the scene geometry
                ray = si.spawn_ray(si.to_world(bs.wo));
                SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray);
                emitter = si_bsdf.emitter(scene);

                // Hit emitter after BSDF sampling
                if (emitter) {

                    // Apply the same reasoning as for the emitter sampling above
                    bool r_t_valid = (caustic_eta == 1.f && has_flag(bs.sampled_type, BSDFFlags::Reflection)) ||
                                     (caustic_eta != 1.f && has_flag(bs.sampled_type, BSDFFlags::Transmission));
                    bool glint_case = (depth == 1 && !m_disable_reflected_emitters);
                    bool valid_connection = on_caustic_caster && ((from_caustic_receiver && r_t_valid) || glint_case);

                    if (!on_caustic_caster || valid_connection) {
                        // Evaluate the emitter for that direction
                        Spectrum emitter_val = emitter->eval(si_bsdf);

                        /* Determine probability of having sampled that same
                           direction using emitter sampling. */
                        DirectionSample3f ds(si_bsdf, si);
                        ds.object = emitter;
                        Float emitter_pdf = select(!has_flag(bs.sampled_type, BSDFFlags::Delta),
                                                   scene->pdf_emitter_direction(si, ds),
                                                   0.f);
                        Float mis = mis_weight(bs.pdf, emitter_pdf);
                        result += mis * throughput * emitter_val;
                    }
                }

                si_prev = si;
                si = std::move(si_bsdf);
            }

            return { result, valid_ray };
        }
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("SingleScatterFilteredPathIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()
protected:
    bool m_disable_reflected_emitters;      // Potentially disable directly reflected emitters / "glints"
};

MTS_IMPLEMENT_CLASS_VARIANT(SingleScatterFilteredPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SingleScatterFilteredPathIntegrator, "Single-Bounce Filtered Path Tracer integrator");
NAMESPACE_END(mitsuba)

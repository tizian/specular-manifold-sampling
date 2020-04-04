#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

#include <mitsuba/render/manifold_glints.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * This is mostly a standard path tracer, but adds the glints variation of
 * specular manifold sampling (SMS), MISed with standard BSDF sampling.
 *
 * This integrator is used for the glint results in the paper, in particular
 * Figures 12 and 19.
 *
 * The part of the glints which is at the moment a bit experimental is the
 * support for blended BSDFs, e.g. to add some non-glinty diffuse component with
 * some colored reflectance. This is not supported with this integrator.
 * The core issue here is that the glints are now evaluated inside the
 * integrator and not as part of a BSDF anymore. A cleaner implementation might
 * require additional API changes in the rest of the system.
 *
 */
template <typename Float, typename Spectrum>
class GlintSMSPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium)
    using SpecularManifold = SpecularManifold<Float, Spectrum>;
    using SpecularManifoldGlints = SpecularManifoldGlints<Float, Spectrum>;

    GlintSMSPathIntegrator(const Properties &props) : Base(props) {
        m_sms_config = SMSConfig();
        m_sms_config.biased                 = props.bool_("biased", false);
        m_sms_config.step_scale             = props.float_("step_scale", 1.f);
        m_sms_config.max_iterations         = props.int_("max_iterations", 20);
        m_sms_config.solver_threshold       = props.float_("solver_threshold", 1e-5f);
        m_sms_config.uniqueness_threshold   = props.float_("uniqueness_threshold", 1e-4f);
        m_sms_config.max_trials             = props.int_("max_trials", -1);
        m_sms_config.bsdf_strategy_only     = props.bool_("bsdf_strategy_only", false);
        m_sms_config.sms_strategy_only      = props.bool_("sms_strategy_only", false);
    }

    bool render(Scene *scene, Sensor *sensor) override {
        bool result = MonteCarloIntegrator<Float, Spectrum>::render(scene, sensor);
        SpecularManifoldGlints::print_statistics();
        return result;
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

                // --------------- Specular Manifold Sampling -----------------

                bool on_glinty_shape = si.shape->is_glinty() && depth == 1;

                if (on_glinty_shape) {
                    SpecularManifoldGlints mf(scene, m_sms_config);
                    result += mf.specular_manifold_sampling(ray.o, si, sampler);
                }

                // --------------------- Emitter sampling ---------------------

                BSDFContext ctx;
                BSDFPtr bsdf = si.bsdf(ray);

                /* As usual, emitter sampling only makes sense on Smooth BSDFs
                   that can be evaluated.
                   Additionally, we already accounted for this direct light
                   contribution with the SMS strategy above in case of glints. */
                if (has_flag(bsdf->flags(), BSDFFlags::Smooth) && !on_glinty_shape) {
                    auto [ds, emitter_weight] = scene->sample_emitter_direction(si, sampler->next_2d(), true);
                    if (ds.pdf != 0.f) {
                        // Query the BSDF for that emitter-sampled direction
                        Vector3f wo = si.to_local(ds.d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo);
                        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                        // Determine density of sampling that same direction using BSDF sampling
                        Float bsdf_pdf = bsdf->pdf(ctx, si, wo);

                        Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                        result += mis * throughput * bsdf_val * emitter_weight;
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

                /* Hit emitter after BSDF sampling. For the glints, we already
                   accounted for this in the SMS step above. */
                if (emitter && !on_glinty_shape) {
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

                si = std::move(si_bsdf);
            }

            return { result, valid_ray };
        }
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("GlintSMSPathIntegrator[\n"
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
    SMSConfig m_sms_config;
};

MTS_IMPLEMENT_CLASS_VARIANT(GlintSMSPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GlintSMSPathIntegrator, "Glint SMS Path Tracer integrator");
NAMESPACE_END(mitsuba)

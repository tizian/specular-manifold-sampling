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
 * This is mostly a standard path tracer, but augmented with the specular
 * manifold sampling (SMS) strategy for (single-bounce) caustics from (near)
 * delta reflections or refractions.
 *
 * Higher-order caustics (e.g. with multiple bounces) are disabled.
 *
 * This integrator is used for many results in the paper, in particular:
 * Figures 4, 6, 8, 9, 14, 15, 16, 17.
 *
 * The integration into the path tracer is more or less specialized for those
 * scenes currently. Extensions such as (approximative) MIS to robustly transition
 * to classical sampling strategies in cases of high roughness or large light
 * sources are an interesting direction for future work.
 *
 *
 * Several options are available to fine-tune which variation of SMS should be
 * performed. See the SMSConfig struct in 'manifold.h'.
 *
 * We can also compare against Manifold Next Event Estimation (MNEE) with it
 * by setting sms_config.mnee_init=true, sms_config.biased=true,
 * sms_config.max_trials=1, sms_config.halfvector_constraints=true, and
 * biased_mnee=true/false depending on which flavour of MNEE should be used.
 *
 */
template <typename Float, typename Spectrum>
class SingleScatterSMSPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium)
    using SpecularManifold = SpecularManifold<Float, Spectrum>;
    using SpecularManifoldSingleScatter = SpecularManifoldSingleScatter<Float, Spectrum>;

    SingleScatterSMSPathIntegrator(const Properties &props) : Base(props) {
        m_sms_config = SMSConfig();
        m_sms_config.biased                 = props.bool_("biased", false);
        m_sms_config.twostage               = props.bool_("twostage", false);
        m_sms_config.halfvector_constraints = props.bool_("halfvector_constraints", false);
        m_sms_config.mnee_init              = props.bool_("mnee_init", false);
        m_sms_config.step_scale             = props.float_("step_scale", 1.f);
        m_sms_config.max_iterations         = props.int_("max_iterations", 20);
        m_sms_config.solver_threshold       = props.float_("solver_threshold", 1e-5f);
        m_sms_config.uniqueness_threshold   = props.float_("uniqueness_threshold", 1e-4f);
        m_sms_config.max_trials             = props.int_("max_trials", -1);

        m_disable_reflected_emitters   = props.bool_("disable_reflected_emitters", false);
        m_biased_mnee                  = props.bool_("biased_mnee", false);
    }

    bool render(Scene *scene, Sensor *sensor) override {
        bool result = MonteCarloIntegrator<Float, Spectrum>::render(scene, sensor);
        SpecularManifoldSingleScatter::print_statistics();
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

                // --------------- Specular Manifold Sampling -----------------

                bool on_caustic_caster = si.shape->is_caustic_caster_single_scatter();

                if (si.shape->is_caustic_receiver() && !on_caustic_caster &&
                    (m_max_depth < 0 || depth + 1 < m_max_depth)) {
                    SpecularManifoldSingleScatter mf(scene, m_sms_config);
                    result += throughput * mf.specular_manifold_sampling(si, sampler);
                }

                // --------------------- Emitter sampling ---------------------

                BSDFContext ctx;
                ctx.sampler = sampler;
                BSDFPtr bsdf = si.bsdf(ray);

                /* As usual, emitter sampling only makes sense on Smooth BSDFs
                   that can be evaluated.
                   Additionally, filter out:
                    - paths that we could previously sample with SMS
                    - paths that are even harder to sample, e.g. paths bouncing
                      off several caustic casters before hitting the light.
                   As a result, we only do emitter sampling on non-caustic
                   casters---with the exception of the first bounce where we might
                   see a direct (glossy) reflection of a light source this way.

                   Note: of course, SMS might not always be the optimal sampling
                   strategy. For example, when rough surfaces are involved it
                   would be still better to do emitter sampling.
                   A way of incoorporating MIS with all of this would be super
                   useful. */
                if (has_flag(bsdf->flags(), BSDFFlags::Smooth) &&
                    !on_caustic_caster || (depth == 1 && !m_disable_reflected_emitters)) {
                    /* In case we didn't scatter off a caustic receiver before
                       or aren't interacting with a caustic caster now, do
                       emitter sampling as usual. */
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

                // Hit emitter after BSDF sampling
                if (emitter) {
                    /* With the same reasoning as in the emitter sampling case,
                       filter out some of the light paths here.
                       Again, this is unfortunately not robust in all cases,
                       for large light sources, BSDF sampling would be more
                       appropriate than relying purely on SMS. */
                    if (!on_caustic_caster || (depth == 1 && !m_disable_reflected_emitters)) {
                        /* Only do BSDF sampling in usual way if we don't interact
                           with a caustic caster now. */

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
                    } else if (m_sms_config.mnee_init && !m_biased_mnee &&
                               si_prev.is_valid() && si_prev.shape->is_caustic_receiver() &&
                               emitter->is_caustic_emitter_single_scatter()) {
                        /* These are the light paths that can be sampled with SMS.
                           In case we're doing MNEE, only a single deterministic path
                           can be generated though, and if we wish to stay unbiased
                           we need to do an additional test here to see if MNEE could
                           generate the currently found light connection as well.
                           Note: Hanika et al. 2015 discuss a more advanced MIS strategy
                           here that also accounts for the smooth probablility density
                           from rough BSDFs. This could be added as well here. */
                        ShapePtr specular_shape = si.shape;
                        EmitterInteraction ei = SpecularManifold::emitter_interaction(scene, si_prev, si_bsdf);

                        SpecularManifoldSingleScatter mf(scene, m_sms_config);
                        auto [success, si_mnee, unused1] = mf.sample_path(specular_shape, si_prev, ei, sampler);
                        if (success) {
                            Vector3f direction_pt   = normalize(si.p - si_prev.p),
                                     direction_mnee = normalize(si_mnee.p - si_prev.p);
                            if (abs(dot(direction_pt, direction_mnee) - 1.f) >= m_sms_config.uniqueness_threshold) {
                                success = false;
                            }
                        }

                        if (!success) {
                            /* MNEE could not find this path, so add it now.
                               There is no MIS needed as we filtered out this class of paths
                               in the emitter sampling strategy above. Note that the original
                               paper about MNEE is more thorough here and does full MIS which
                               improves the case of caustics from rough BSDFs. For simplicity
                               we leave this out, but it could be added as well. In that
                               case, we would also need to perform this "MNEE check" in the
                               emitter sampling step above. */
                            result += throughput * emitter->eval(si_bsdf);
                        }
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
        return tfm::format("SingleScatterSMSPathIntegrator[\n"
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
    bool m_disable_reflected_emitters;      // Potentially disable directly reflected emitters / "glints"
    bool m_biased_mnee;                     // Make MNEE biased by filtering out caustic paths that can't be sampled with it
};

MTS_IMPLEMENT_CLASS_VARIANT(SingleScatterSMSPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SingleScatterSMSPathIntegrator, "Single-Bounce SMS Path Tracer integrator");
NAMESPACE_END(mitsuba)

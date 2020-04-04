#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

#include <mitsuba/render/manifold_ms.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * This is mostly a standard path tracer, but augmented with the specular
 * manifold sampling (SMS) strategy for (multi-bounce) caustics from (near)
 * delta reflections or refractions.
 *
 * This integrator is used for Figure 18 in the paper.
 *
 * The integration into the path tracer is more or less specialized for that
 * scene currently. Extensions such as (approximative) MIS to robustly transition
 * to classical sampling strategies in cases of high roughness or large light
 * sources are an interesting direction for future work.
 *
 * Several options are available to fine-tune which variation of SMS should be
 * performed. See the SMSConfig struct in 'manifold.h'. The number of bounces
 * for the caustics should also be set there explicitly.
 *
 * We can also compare against Manifold Next Event Estimation (MNEE) with it
 * by setting sms_config.mnee_init=true, sms_config.biased=true,
 * sms_config.max_trials=1, sms_config.halfvector_constraints=true, and
 * biased_mnee=true/false depending on which flavour of MNEE should be used.
 *
 * This multi-bounce implementation doesn't support specular manifold sampling
 * with glossy/rough materials, but it could be extended to that case as well.
 *
 */
template <typename Float, typename Spectrum>
class MultiScatterSMSPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium)
    using SpecularManifold = SpecularManifold<Float, Spectrum>;
    using SpecularManifoldMultiScatter = SpecularManifoldMultiScatter<Float, Spectrum>;

protected:
    /* The integration of SMS is pretty straight forward in the multi-bounce
       case as well---but Manifold Next Event Estimation unfortunately gets
       a bit more tricky to implement.
       For each random light connection we want to check if that light path could
       have also been generated with MNEE, so we need to keep some additional
       state around. This helper struct here takes care of this. */
    struct MNEEHelper {
        enum MNEEState {
            Default = 0,
            HitReceiver,
            HitCaster,
            HitEmitter,
        } state;
        int bounce;

        const Scene *scene = nullptr;
        SMSConfig sms_config;

        SurfaceInteraction3f si_endpoint;
        std::vector<ShapePtr> specular_shapes;
        std::vector<Point3f>  specular_positions;

        void init(const Scene *s, const SMSConfig &config) {
            scene = s;
            sms_config = config;
            reset();
        }

        void reset() {
            state = MNEEState::Default;
            specular_shapes.clear();
            specular_positions.clear();
            bounce = -1;
        }

        bool is_possible() const {
            return state == MNEEState::HitEmitter && bounce == sms_config.bounces;
        }

        void state_transition(const SurfaceInteraction3f &next_si) {
            // In any case, when we hit a caustic receiver we start from scratch
            if (next_si.is_valid() &&
                next_si.shape->is_caustic_receiver()) {
                reset();
                state = MNEEState::HitReceiver;
                // Save this interaction for later
                si_endpoint = next_si;
                return;
            }

            EmitterPtr emitter = next_si.emitter(scene);

            if (state == MNEEState::HitReceiver) {
                // From here we should hit the first caustic caster
                if (next_si.is_valid() &&
                    next_si.shape->is_caustic_caster_multi_scatter()) {
                    // Record hit and transition to HitCaster
                    specular_shapes.push_back(next_si.shape);
                    specular_positions.push_back(next_si.p);
                    bounce = 1;
                    state = MNEEState::HitCaster;
                } else {
                    reset();
                }
                return;
            } else if (state == MNEEState::HitCaster) {
                /* From here we can hit either caustic caster or bouncer to
                   build up the specular chain
                   or
                   hit a light source and complete a potential MNEE path. */
                if (bounce < sms_config.bounces &&
                    next_si.is_valid() &&
                    (next_si.shape->is_caustic_caster_multi_scatter() ||
                     next_si.shape->is_caustic_bouncer())) {
                    // Record hit but stay in this state
                    specular_shapes.push_back(next_si.shape);
                    specular_positions.push_back(next_si.p);
                    bounce++;
                } else if (bounce == sms_config.bounces &&
                           emitter && emitter->is_caustic_emitter_multi_scatter()) {
                    state = MNEEState::HitEmitter;
                } else {
                    reset();
                }
                return;
            } else if (state == MNEEState::HitEmitter) {
                // We completed a path. Reset now as the path was processed in the meantime.
                reset();
                return;
            } else {
                // Any other case, e.g. miss the scene
                reset();
                return;
            }
        }
    };

    static inline ThreadLocal<SpecularManifoldMultiScatter> tl_manifold{};
    static inline ThreadLocal<MNEEHelper> tl_mnee{};

public:
    MultiScatterSMSPathIntegrator(const Properties &props) : Base(props) {
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

        m_sms_config.bounces                = props.int_("bounces", 2);

        m_biased_mnee                  = props.bool_("biased_mnee", false);
    }

    bool render(Scene *scene, Sensor *sensor) override {
        bool result = MonteCarloIntegrator<Float, Spectrum>::render(scene, sensor);
        SpecularManifoldMultiScatter::print_statistics();
        return result;
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        auto &mf = (SpecularManifoldMultiScatter &)tl_manifold;
        mf.init(scene, m_sms_config);
        auto &mnee = (MNEEHelper &)tl_mnee;
        mnee.init(scene, m_sms_config);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return { 0.f, 0.f };
        } else {
            RayDifferential3f ray = ray_;
            Float eta = 1.f;
            Spectrum throughput(1.f), result(0.f);
            bool specular_camera_path = true;   // To capture emitters visible direcly through purely specular reflection/refractions

            // ---------------------- First intersection ----------------------

            SurfaceInteraction3f si = scene->ray_intersect(ray);
            Mask valid_ray = si.is_valid();
            EmitterPtr emitter = si.emitter(scene);
            // Keep track of state regarding previous bounces in order to do unbiased MNEE
            mnee.state_transition(si);

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

                bool on_caustic_caster = si.shape->is_caustic_caster_multi_scatter() ||
                                         si.shape->is_caustic_bouncer();

                if (si.shape->is_caustic_receiver() && !on_caustic_caster &&
                    (m_max_depth < 0 || depth + m_sms_config.bounces < m_max_depth)) {
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
                    !on_caustic_caster) {
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
                if (!has_flag(bs.sampled_type, BSDFFlags::Delta)) {
                    specular_camera_path = false;
                }

                if (all(eq(throughput, 0.f)))
                    break;

                // Intersect the BSDF ray against the scene geometry
                ray = si.spawn_ray(si.to_world(bs.wo));
                SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray);
                emitter = si_bsdf.emitter(scene);

                // Keep track of state regarding previous bounces in order to do unbiased MNEE
                mnee.state_transition(si_bsdf);

                // Hit emitter after BSDF sampling
                if (emitter) {
                    /* With the same reasoning as in the emitter sampling case,
                       filter out some of the light paths here.
                       Again, this is unfortunately not robust in all cases,
                       for large light sources, BSDF sampling would be more
                       appropriate than relying purely on SMS. */
                    if (!on_caustic_caster || specular_camera_path) {
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
                               mnee.is_possible()) {
                        /* These are the light paths that can be sampled with SMS.
                           In case we're doing MNEE, only a single deterministic path
                           can be generated though, and if we wish to stay unbiased
                           we need to do an additional test here to see if MNEE could
                           generate the currently found light connection as well.
                           Note: Hanika et al. 2015 discuss a more advanced MIS strategy
                           here that also accounts for the smooth probablility density
                           from rough BSDFs. This could be added as well here. To
                           support the rough case properly, the sampled half-vectors
                           of specular paths to be tested with MNEE would need to be
                           passed to the SpecularManifoldMultiScatter datastructure
                           somehow. */

                        ShapePtr specular_shape = mnee.specular_shapes[0];
                        EmitterInteraction ei = SpecularManifold::emitter_interaction(scene, mnee.si_endpoint, si_bsdf);
                        bool success = mf.sample_path(specular_shape, mnee.si_endpoint, ei, sampler, true);
                        if (success) {
                            auto current_path = mf.current_path();
                            for (size_t k = 0; k < current_path.size(); ++k) {
                                Point3f p_pt = mnee.specular_positions[k],
                                        p_mnee = current_path[k].p;
                                if (norm(p_pt - p_mnee) >= 1e-5f) {
                                    success = false;
                                }
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

                si = std::move(si_bsdf);
            }

            return { result, valid_ray };
        }
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("MultiScatterSMSPathIntegrator[\n"
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
    bool m_biased_mnee;      // Make MNEE biased by filtering out caustic paths that can't be sampled with it
};

MTS_IMPLEMENT_CLASS_VARIANT(MultiScatterSMSPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MultiScatterSMSPathIntegrator, "Multi-Bounce SMS Path Tracer integrator");
NAMESPACE_END(mitsuba)

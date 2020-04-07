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
 * This is a mostly a standard path tracer, with the exception that some of the
 * caustic light paths are filtered out.
 *
 * More specifically, it produces the same results as 'path_sms_teaser' but
 * relies on standard BSDF / emitter sampling strategies to find the castics
 * that are sampled with specular manifold sampling in the other one. It's
 * purpose are fair comparisons of our technique in a setting where noise from
 * other (even more challenging light paths) is not present.
 *
 * Glints are not supported in this integrator.
 *
 * It is used for Figure 13 in the paper.
 *
 */
template <typename Float, typename Spectrum>
class TeaserFilteredPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium)

    enum PathState {
        Default = 0,
        HitReceiver,
        HitSingleCaster,
        HitSingleEmitter,
        HitMultiCaster,
        HitMultiEmitter,
    };

public:
    TeaserFilteredPathIntegrator(const Properties &props) : Base(props) {
        m_multi_caustic_bounces = props.int_("multi_caustic_bounces", 2);
        m_non_sms_paths_enabled = props.bool_("non_sms_paths_enabled", true);
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
            PathRecord path_record(scene, m_multi_caustic_bounces);

            RayDifferential3f ray = ray_;
            Float eta = 1.f;
            Spectrum throughput(1.f), result(0.f);

            // ---------------------- First intersection ----------------------

            SurfaceInteraction3f si = scene->ray_intersect(ray);
            Mask valid_ray = si.is_valid();
            EmitterPtr emitter = si.emitter(scene);
            // Keep track of state regarding previous bounces
            path_record.state_transition(si, si, BSDFSample3f());

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
                                         si.shape->is_caustic_caster_multi_scatter() ||
                                         si.shape->is_caustic_bouncer();

                // --------------------- Emitter sampling ---------------------

                BSDFContext ctx;
                BSDFPtr bsdf = si.bsdf(ray);

                if (has_flag(bsdf->flags(), BSDFFlags::Smooth)) {
                    DirectionSample3f ds;
                    Spectrum emitter_weight;

                    /* Whenever we are on a caustic caster, only certain light
                       connections are valid. This is summarized in the
                       'sample_filtered_emitter_direction' method below. */
                    if (!on_caustic_caster) {
                        std::tie(ds, emitter_weight) = scene->sample_emitter_direction(si, sampler->next_2d(), true);
                    } else {
                        std::tie(ds, emitter_weight) = sample_filtered_emitter_direction(scene, si, sampler->next_2d(),
                                                                                         path_record.state, path_record.bounce);
                    }

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
                           we are sampling with SMS. */
                        if (on_caustic_caster) {
                            Complex<Spectrum> caustic_ior = si.bsdf()->ior(si);
                            Float caustic_eta = select(all(eq(0.f, imag(caustic_ior))), hmean(real(caustic_ior)), 1.f);
                            Float cos_theta_x = Frame3f::cos_theta(wo),
                                  cos_theta_y = Frame3f::cos_theta(si.wi);
                            bool refraction = cos_theta_x * cos_theta_y < 0.f;
                            bool reflection = !refraction;
                            if ((caustic_eta == 1.f && !reflection) ||
                                (caustic_eta != 1.f && !refraction)) {
                                mis = 0.f;
                            }
                        }

                        // If requested, also filter out all non-SMS paths
                        if (!on_caustic_caster && !m_non_sms_paths_enabled) {
                            mis = 0.f;
                        }

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

                // Keep track of state regarding previous bounces
                path_record.state_transition(si, si_bsdf, bs);

                /* With the same reasoning as in the emitter sampling case,
                   filter out some of the light paths here. */
                if (on_caustic_caster &&
                    !filtered_emitter_hit_valid(emitter, path_record.state, path_record.bounce)) {
                    emitter = nullptr;
                }

                // If requested, also filter out all non-SMS paths
                if (!on_caustic_caster && !m_non_sms_paths_enabled) {
                    emitter = nullptr;
                }

                // Hit emitter after BSDF sampling
                if (emitter) {
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

    /// Sample direct lighting contribution from a caustic caster shape. Some filtering is necessary here.
    std::tuple<DirectionSample3f, Spectrum>
    sample_filtered_emitter_direction(const Scene *scene,
                                      const Interaction3f &ref,
                                      const Point2f &sample_,
                                      PathState state, int bounce,
                                      Mask active = true) const {

        using EmitterPtr = replace_scalar_t<Float, Emitter*>;

        Point2f sample(sample_);
        DirectionSample3f ds = zero<DirectionSample3f>();
        Spectrum spec = 0.f;

        bool single_emitter = (state == PathState::HitSingleCaster && bounce == 1);
        bool multi_emitter  = (state == PathState::HitMultiCaster && bounce == m_multi_caustic_bounces);
        bool no_emitter     = !single_emitter && !multi_emitter;
        if (no_emitter) {
            return { ds, spec };
        }

        const auto emitters = (single_emitter ? scene->caustic_emitters_single_scatter()
                                              : scene->caustic_emitters_multi_scatter());

        if (likely(!emitters.empty())) {
            if (emitters.size() == 1) {
                // Fast path if there is only one emitter
                std::tie(ds, spec) = emitters[0]->sample_direction(ref, sample, active);
            } else {
                ScalarFloat emitter_pdf = 1.f / emitters.size();

                // Randomly pick an emitter
                UInt32 index = min(UInt32(sample.x() * (ScalarFloat) emitters.size()), (uint32_t) emitters.size()-1);

                // Rescale sample.x() to lie in [0,1) again
                sample.x() = (sample.x() - index*emitter_pdf) * emitters.size();

                EmitterPtr emitter = gather<EmitterPtr>(emitters.data(), index, active);

                // Sample a direction towards the emitter
                std::tie(ds, spec) = emitter->sample_direction(ref, sample, active);

                // Account for the discrete probability of sampling this emitter
                ds.pdf *= emitter_pdf;
                spec *= rcp(emitter_pdf);
            }

            active &= neq(ds.pdf, 0.f);

            // Perform a visibility test if requested
            Ray3f ray(ref.p, ds.d, math::RayEpsilon<Float> * (1.f + hmax(abs(ref.p))),
                      ds.dist * (1.f - math::ShadowEpsilon<Float>), ref.time, ref.wavelengths);
            if (scene->ray_test(ray, active)) {
                spec = 0.f;
            }
        }
        return { ds, spec };
    }

    bool filtered_emitter_hit_valid(const EmitterPtr emitter,
                                    PathState state, int bounce) const {
        if (state == PathState::HitSingleEmitter &&
            bounce == 1 &&
            emitter->is_caustic_emitter_multi_scatter())
            return true;

        if (state == PathState::HitMultiEmitter &&
            bounce == m_multi_caustic_bounces &&
            emitter->is_caustic_emitter_multi_scatter())
            return true;

        return false;
    }

    std::string to_string() const override {
        return tfm::format("TeaserFilteredPathIntegrator[\n"
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
    int m_multi_caustic_bounces;
    bool m_non_sms_paths_enabled;

    struct PathRecord {
        PathState state;
        int bounce;
        int max_bounces;

        const Scene *scene = nullptr;

        PathRecord(const Scene *s, int b) {
            scene = s;
            max_bounces = b;
            reset();
        }

        void reset() {
            state = PathState::Default;
            bounce = -1;
        }

        void state_transition(const SurfaceInteraction3f &current_si,
                              const SurfaceInteraction3f &next_si,
                              const BSDFSample3f &bs) {
            // In any case, when we hit a caustic receiver we start from scratch
            if (next_si.is_valid() &&
                next_si.shape->is_caustic_receiver()) {
                reset();
                state = PathState::HitReceiver;
                return;
            }

            EmitterPtr emitter = next_si.emitter(scene);

            if (state == PathState::HitReceiver) {
                // From here we should hit the first caustic caster
                if (next_si.is_valid() &&
                    next_si.shape->is_caustic_caster_single_scatter()) {
                    // Record hit and transition
                    bounce = 1;
                    state = PathState::HitSingleCaster;
                } else if (next_si.is_valid() &&
                           next_si.shape->is_caustic_caster_multi_scatter()) {
                    // Record hit and transition
                    bounce = 1;
                    state = PathState::HitMultiCaster;
                } else {
                    reset();
                }
            } else if (state == PathState::HitSingleCaster) {
                // Did we reflect / refract correctly?
                Complex<Spectrum> ior = current_si.bsdf()->ior(current_si);
                Float eta = select(all(eq(0.f, imag(ior))), hmean(real(ior)), 1.f);
                if ((eta == 1.f && has_flag(bs.sampled_type, BSDFFlags::Transmission)) ||
                    (eta != 1.f && has_flag(bs.sampled_type, BSDFFlags::Reflection))) {
                    reset();
                } else {
                    // Did we hit the right light class?
                    if (bounce == 1 &&
                        emitter && emitter->is_caustic_emitter_single_scatter()) {
                        // We completed the light path successfully
                        state = PathState::HitSingleEmitter;
                    } else {
                        // Fail
                        reset();
                    }

                }
            } else if (state == PathState::HitMultiCaster) {

                // Did we reflect / refract correctly?
                Complex<Spectrum> ior = current_si.bsdf()->ior(current_si);
                Float eta = select(all(eq(0.f, imag(ior))), hmean(real(ior)), 1.f);
                if ((eta == 1.f && has_flag(bs.sampled_type, BSDFFlags::Transmission)) ||
                    (eta != 1.f && has_flag(bs.sampled_type, BSDFFlags::Reflection))) {
                    reset();
                } else {

                    // Did we hit another possible caustic caster/bouncer?
                    if (bounce < max_bounces &&
                        next_si.is_valid() &&
                        (next_si.shape->is_caustic_caster_multi_scatter() ||
                         next_si.shape->is_caustic_bouncer())) {
                        // Record the hit and stay in this state
                        bounce++;
                    } else if (bounce == max_bounces &&
                               emitter && emitter->is_caustic_emitter_multi_scatter()) {
                        // We completed the light path successfully
                        state = PathState::HitMultiEmitter;
                    } else {
                        // Fail
                        reset();
                    }

                }

            } else if (state == PathState::HitSingleEmitter ||
                       state == PathState::HitMultiEmitter) {
                // We completed a path. Reset now as the path was processed in the meantime.
                reset();
            } else {
                // Any other case, e.g. miss the scene
                reset();
            }
        }
    };

};

MTS_IMPLEMENT_CLASS_VARIANT(TeaserFilteredPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(TeaserFilteredPathIntegrator, "Teaser Filtered Path Tracer integrator");
NAMESPACE_END(mitsuba)

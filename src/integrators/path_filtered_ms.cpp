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
 * More specifically, it produces the same results as 'path_sms_ms' but relies
 * on standard BSDF / emitter sampling strategies to find the castics that
 * are sampled with specular manifold sampling in the other one. It's purpose
 * are fair comparisons of our technique in a setting where noise from other
 * (even more challenging light paths) is not present.
 *
 * It is used for Figure 18 in the paper.
 *
 */
template <typename Float, typename Spectrum>
class MultiScatterFilteredPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium)

public:
    MultiScatterFilteredPathIntegrator(const Properties &props) : Base(props) {
        m_bounces = props.int_("bounces", 2);   // Number of specular caustic bounces.
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
            PathRecord path_record(scene, m_bounces);

            RayDifferential3f ray = ray_;
            Float eta = 1.f;
            Spectrum throughput(1.f), result(0.f);
            bool specular_camera_path = true;   // To capture emitters visible direcly through purely specular reflection/refractions

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

                bool on_caustic_caster = si.shape->is_caustic_caster_multi_scatter() ||
                                         si.shape->is_caustic_bouncer();

                // --------------------- Emitter sampling ---------------------

                BSDFContext ctx;
                BSDFPtr bsdf = si.bsdf(ray);

                if (has_flag(bsdf->flags(), BSDFFlags::Smooth)) {

                    bool valid_connection = path_record.light_connection_valid();
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
                               we are sampling with SMS. */
                            if (valid_connection) {
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
                if (!has_flag(bs.sampled_type, BSDFFlags::Delta)) {
                    specular_camera_path = false;
                }

                if (all(eq(throughput, 0.f)))
                    break;

                // Intersect the BSDF ray against the scene geometry
                ray = si.spawn_ray(si.to_world(bs.wo));
                SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray);
                emitter = si_bsdf.emitter(scene);

                // Keep track of state regarding previous bounces
                path_record.state_transition(si, si_bsdf, bs);

                // Hit emitter after BSDF sampling
                if (emitter) {

                    if (!on_caustic_caster || path_record.light_hit_valid() || specular_camera_path) {
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

                si = std::move(si_bsdf);
            }

            return { result, valid_ray };
        }
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("MultiScatterFilteredPathIntegrator[\n"
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
    int m_bounces;

    struct PathRecord {
        enum PathState {
            Default = 0,
            HitReceiver,
            HitCaster,
            HitEmitter,
        } state;
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

        bool light_hit_valid() const {
            return state == PathState::HitEmitter && bounce == max_bounces;
        }

        bool light_connection_valid() const {
            return state == PathState::HitCaster && bounce == max_bounces;
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
                    next_si.shape->is_caustic_caster_multi_scatter()) {
                    // Record hit and transition to HitCaster
                    bounce = 1;
                    state = PathState::HitCaster;
                } else {
                    reset();
                }
                return;
            } else if (state == PathState::HitCaster) {
                /* From here we can hit either caustic caster or bouncer to
                   build up the specular chain
                   or
                   hit a light source and complete a light path. */
                if (bounce < max_bounces &&
                    next_si.is_valid() &&
                    (next_si.shape->is_caustic_caster_multi_scatter() ||
                     next_si.shape->is_caustic_bouncer())) {
                    /* We continue with the specular path, but need to make
                       sure we sampled the correct scattering type. */

                    Complex<Spectrum> ior = current_si.bsdf()->ior(current_si);
                    Float eta = select(all(eq(0.f, imag(ior))), hmean(real(ior)), 1.f);
                    if ((eta == 1.f && has_flag(bs.sampled_type, BSDFFlags::Transmission)) ||
                        (eta != 1.f && has_flag(bs.sampled_type, BSDFFlags::Reflection))) {
                        reset();
                    } else {
                        // Record the hit and stay in this state
                        bounce++;
                    }
                } else if (bounce == max_bounces &&
                           emitter && emitter->is_caustic_emitter_multi_scatter()) {
                    state = PathState::HitEmitter;
                } else {
                    reset();
                }
                return;
            } else if (state == PathState::HitEmitter) {
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

};

MTS_IMPLEMENT_CLASS_VARIANT(MultiScatterFilteredPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MultiScatterFilteredPathIntegrator, "Multi-Bounce Filtered Path Tracer integrator");
NAMESPACE_END(mitsuba)

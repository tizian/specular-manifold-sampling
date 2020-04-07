#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

#include <mitsuba/render/manifold_ss.h>
#include <mitsuba/render/manifold_ms.h>
#include <mitsuba/render/manifold_glints.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * This is mostly a standard path tracer, but augmented with the specular
 * manifold sampling (SMS) strategy for single-bounce, multi-bounce caustics and
 * glints.
 *
 * It is essentially specialized to the "shop window" teaser image scene
 * from the paper (Figure 1 and 13).
 *
 * It uses the glint method for the shoes, single-bounce caustics from the
 * conductor pedestals, and two bounce caustics involving the two-sided dielectric
 * pedestal.
 *
 * Higher order scattering, e.g. between the individual pedestals is disabled.
 *
 * Additional complications comes from the glint integration:
 *
 * 1) To show glints behind the double-sided glass window, we trace ray differentials
 *    through delta refractions. We also add a "hacky" light connection strategy
 *    that passes through the glass, assuming the direction after the two subsequent
 *    refractions is unchanged. (This is simply to connect to the envmap around the
 *    scene.)
 *
 * 2) Two of the glinty pairs of shoes have an additional diffuse BSDF blended
 *    with the glint component. This means we have to selectively evaluate only
 *    specific components of a potential glinty 'blendbsdf' plugin in parts of
 *    the integrator.
 *
 */
template <typename Float, typename Spectrum>
class TeaserSMSPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, SensorPtr, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium)
    using EmitterInteraction            = EmitterInteraction<Float, Spectrum>;
    using SpecularManifold              = SpecularManifold<Float, Spectrum>;
    using SpecularManifoldSingleScatter = SpecularManifoldSingleScatter<Float, Spectrum>;
    using SpecularManifoldMultiScatter  = SpecularManifoldMultiScatter<Float, Spectrum>;
    using SpecularManifoldGlints        = SpecularManifoldGlints<Float, Spectrum>;

    static inline ThreadLocal<SpecularManifoldMultiScatter> tl_manifold_ms{};

    TeaserSMSPathIntegrator(const Properties &props) : Base(props) {

        m_caustics_enabled                           = props.bool_("caustics_enabled", true);
        m_sms_config_caustics                        = SMSConfig();
        m_sms_config_caustics.biased                 = props.bool_("caustics_biased", false);
        m_sms_config_caustics.twostage               = props.bool_("caustics_twostage", false);
        m_sms_config_caustics.halfvector_constraints = props.bool_("caustics_halfvector_constraints", false);
        m_sms_config_caustics.step_scale             = props.float_("caustics_step_scale", 1.f);
        m_sms_config_caustics.max_iterations         = props.int_("caustics_max_iterations", 20);
        m_sms_config_caustics.solver_threshold       = props.float_("caustics_solver_threshold", 1e-5f);
        m_sms_config_caustics.uniqueness_threshold   = props.float_("caustics_uniqueness_threshold", 1e-4f);
        m_sms_config_caustics.max_trials             = props.int_("caustics_max_trials", -1);
        m_sms_config_caustics.bounces                = props.int_("caustics_bounces", 2);

        m_glints_enabled                           = props.bool_("glints_enabled", true);
        m_sms_config_glints                        = SMSConfig();
        m_sms_config_glints.biased                 = props.bool_("glints_biased", false);
        m_sms_config_glints.step_scale             = props.float_("glints_step_scale", 1.f);
        m_sms_config_glints.max_iterations         = props.int_("glints_max_iterations", 20);
        m_sms_config_glints.solver_threshold       = props.float_("glints_solver_threshold", 1e-5f);
        m_sms_config_glints.uniqueness_threshold   = props.float_("glints_uniqueness_threshold", 1e-4f);
        m_sms_config_glints.max_trials             = props.int_("glints_max_trials", -1);

        m_non_sms_paths_enabled = props.bool_("non_sms_paths_enabled", true);
    }

    bool render(Scene *scene, Sensor *sensor) override {
        bool result = MonteCarloIntegrator<Float, Spectrum>::render(scene, sensor);
        SpecularManifoldSingleScatter::print_statistics();
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

        auto &mf_ms = (SpecularManifoldMultiScatter &)tl_manifold_ms;
        mf_ms.init(scene, m_sms_config_caustics);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return { 0.f, 0.f };
        } else {
            RayDifferential3f ray = ray_;
            Point3f sensor_position = ray.o;
            Float eta = 1.f;
            Spectrum throughput(1.f), result(0.f);

            // ---------------------- First intersection ----------------------

            SurfaceInteraction3f si = scene->ray_intersect(ray);
            if (si.is_valid()) {
                si.compute_partials(ray);
            }
            Mask valid_ray = si.is_valid();
            EmitterPtr emitter = si.emitter(scene);

            if (emitter) {
                result += emitter->eval(si);
            }

            // ---------------------- Main loop ----------------------

            int glint_depth = 1;    // Keep track of depth without delta transmissions
            for (int depth = 1;; ++depth) {

                // ------------------ Possibly terminate path -----------------

                if (!si.is_valid())
                    break;

                if (depth > m_rr_depth) {
                    Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                    if (sampler->next_1d() > q)
                        break;
                    throughput *= rcp(q);
                }

                if (uint32_t(depth) >= uint32_t(m_max_depth))
                    break;

                bool on_caustic_receiver = si.shape->is_caustic_receiver();
                bool on_caustic_caster   = si.shape->is_caustic_caster_single_scatter() ||
                                           si.shape->is_caustic_caster_multi_scatter() ||
                                           si.shape->is_caustic_bouncer();
                bool on_glinty_shape     = si.shape->is_glinty() && glint_depth == 1;

                // --------------- Specular Manifold Sampling -----------------

                if (m_caustics_enabled && on_caustic_receiver) {
                    if (m_max_depth < 0 || depth + 1 < m_max_depth) {
                        SpecularManifoldSingleScatter mf_ss(scene, m_sms_config_caustics);
                        result += throughput * mf_ss.specular_manifold_sampling(si, sampler);
                    }

                    if (m_max_depth < 0 || depth + m_sms_config_caustics.bounces < m_max_depth) {
                        result += throughput * mf_ms.specular_manifold_sampling(si, sampler);
                    }
                }

                if (m_glints_enabled && on_glinty_shape) {
                    result += glint_contribution(sensor_position, si, scene, sampler);
                }

                // --------------------- Emitter sampling ---------------------

                BSDFContext ctx;
                BSDFPtr bsdf = si.bsdf(ray);

                /* As usual, emitter sampling only makes sense on Smooth BSDFs
                   that can be evaluated.
                   Additionally, filter out:
                    - paths that we could previously sample with SMS
                    - paths that are even harder to sample, e.g. paths bouncing
                      off several caustic casters before hitting the light.
                   As a result, we only do emitter sampling on non-caustic
                   casters.

                   These remaining paths can further be disabled to only look
                   at SMS compatible paths.*/
                if (has_flag(bsdf->flags(), BSDFFlags::Smooth) && !on_caustic_caster && m_non_sms_paths_enabled) {
                    auto [ds, emitter_weight] = scene->sample_emitter_direction(si, sampler->next_2d(), true);
                    if (ds.pdf != 0.f) {
                        /* We already handled direct light contributions for the glint
                           case separately, so only evaluate remaining (non-glinty)
                           component here. */
                        if (on_glinty_shape) {
                            ctx.component = 1;
                        }

                        // Query the BSDF for that emitter-sampled direction
                        Vector3f wo = si.to_local(ds.d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo);
                        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);
                        ctx.component = -1;

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
                RayDifferential3f rd = si.spawn_ray(si.to_world(bs.wo));
                if (has_flag(bs.sampled_type, BSDFFlags::DeltaTransmission)) {
                    // Propagate ray differentials through delta refraction events.
                    // Based on implementation in pbrt-v3
                    Vector3f wi = si.to_world(si.wi),
                             wo = si.to_world(bs.wo);

                    rd.has_differentials = true;
                    rd.o_x = si.p + si.dp_dx;
                    rd.o_y = si.p + si.dp_dy;
                    Float eta = bs.eta;
                    Vector3f w = -wi;
                    Normal3f n = si.sh_frame.n;
                    if (dot(wi, n) < 0.f) eta = rcp(eta);

                    auto [dn_du, dn_dv] = si.shape->normal_derivative(si, true);
                    Normal3f dn_dx = dn_du * si.duv_dx[0] +
                                     dn_dv * si.duv_dx[1];
                    Normal3f dn_dy = dn_du * si.duv_dy[0] +
                                     dn_dv * si.duv_dy[1];

                    Vector3f dwi_dx = -ray.d_x - wi,
                             dwi_dy = -ray.d_y - wi;

                    Float dDN_dx = dot(dwi_dx, n) + dot(wi, dn_dx),
                          dDN_dy = dot(dwi_dy, n) + dot(wi, dn_dy);

                    Float mu = eta * dot(w, n) - dot(wo, n);
                    Float dmu_dx = (eta - (eta * eta * dot(w, n)) / dot(wo, n)) * dDN_dx,
                          dmu_dy = (eta - (eta * eta * dot(w, n)) / dot(wo, n)) * dDN_dy;

                    rd.d_x = wo + eta * dwi_dx - Vector3f(mu * dn_dx + dmu_dx * n);
                    rd.d_y = wo + eta * dwi_dy - Vector3f(mu * dn_dy + dmu_dy * n);
                }
                ray = rd;
                SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray);
                si_bsdf.compute_partials(ray);
                emitter = si_bsdf.emitter(scene);

                /* With the same reasoning as in the emitter sampling case,
                   filter out some of the light paths here.
                   We also already accounted for the glinty component of this
                   in the glint specific code. */
                if (emitter && !on_caustic_caster && m_non_sms_paths_enabled && !on_glinty_shape) {
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

                // Only increment glint_depth when we don't go through delta dielectrics
                if (!has_flag(bs.sampled_type, BSDFFlags::DeltaTransmission)) {
                    glint_depth++;
                }
            }

            return { result, valid_ray };
        }
    }

    //! @}
    // =============================================================


    /* This is mostly a copy of SpecularManifoldGlints::specular_manifold_sampling
       but with some changes regarding an envmap behind (delta)dielectric windows.
       It is further specialized to only the "biased" SMS strategy for the glints
       and doesn't support surface roughness. */
    Spectrum glint_contribution(const Point3f &sensor_position,
                                const SurfaceInteraction3f &si,
                                const Scene *scene,
                                ref<Sampler> sampler) const {
        ScopedPhase scope_phase(ProfilerPhase::SMSGlints);

        if (unlikely(!si.is_valid() || !si.shape->is_glinty())) {
            return 0.f;
        }

        Spectrum result(0.f);
        SpecularManifoldGlints mf_glints(scene, m_sms_config_glints);

        // Sample emitter interaction
        EmitterInteraction ei = SpecularManifold::sample_emitter_interaction(si, scene->emitters(), sampler);

        // Check visibility to emitter already here, otherwise skip to BSDF sampling strategy
        auto [footprint_visible, emitter_throughput] = emitter_visibility_test(si, ei, scene, sampler);
        if (footprint_visible) {
            Vector3f n_offset(0.f, 0.f, 1.f);

            // ----------------------- SMS Sampling strategy --------------------------

            // Always do the biased SMS version
            Spectrum sms_weight(0.f);
            std::vector<Point2f> solutions;
            for (int m = 0; m < m_sms_config_glints.max_trials; ++m) {
                auto [success, uv_final, unused] = mf_glints.sample_glint(sensor_position, ei, si, sampler, n_offset);
                if (!success) {
                    continue;
                }

                // Check if this is a new and unique solution
                bool duplicate = false;
                for (size_t k = 0; k < solutions.size(); ++k) {
                    if (norm(uv_final - solutions[k]) < m_sms_config_glints.uniqueness_threshold) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) {
                    continue;
                }
                solutions.push_back(uv_final);

                sms_weight += mf_glints.evaluate_glint_contribution(sensor_position, ei, si, uv_final);
            }
            bool sms_success = solutions.size() > 0;

            if (sms_success) {
                // Evaluate the complate path throughput
                SensorPtr sensor = scene->sensors()[0];
                Vector3f sensor_direction = normalize(si.p - sensor_position);
                auto inv_trafo = sensor->camera_to_world(si.time).inverse();
                Float We = sensor->importance(inv_trafo.transform_affine(sensor_position),
                                              inv_trafo * sensor_direction);
                Vector2i film_size = sensor->film()->size();
                We *= film_size[0]*film_size[1];

                /* The true pdf of what we just did with SMS is unkown, but apart from some
                   offset inside the pixel footprint it still closely resembels standard
                   direct illumination / emitter sampling. So we use that as an approximate
                   pdf for MIS. */
                Float p_sms = 1.f;

                Vector3f emitter_direction = ei.p - si.p;
                Float emitter_dist = norm(emitter_direction);
                emitter_direction *= rcp(emitter_dist);

                if (!ei.is_delta()) {
                    DirectionSample3f ds;
                    ds.object = ei.emitter;
                    ds.p = ei.p; ds.n = ei.n; ds.d = ei.d;
                    ds.dist = emitter_dist;
                    p_sms = scene->pdf_emitter_direction(si, ds);
                }

                /* Compute hypothetical pdf for this direction using BSDF sampling for
                   the full directional density of the BSDF averaged over the footprint.
                   This is tricky to evaluate exactly, but we can compute a good
                   approximation based on LEAN mapping. */
                Float p_bsdf = mf_glints.lean_pdf(si, si.to_local(emitter_direction));
                /* Apply a constant offset to make sure the density is non-zero for all
                   parts of the integrand. */
                p_bsdf += 1.f;

                /* Also take into account that the glints might be blended with some
                   other (e.g. diffuse) component. */
                Float blend_weight = si.shape->bsdf()->glint_component_weight(si);

                Float mis = mis_weight(p_sms, p_bsdf);
                result += mis * blend_weight * We * sms_weight * emitter_throughput * ei.weight;
            }
        }

        // ---------------------- BSDF Sampling strategy --------------------------

        // Sample (another) offset UV inside the pixel footprint
        Point2f fp_sample = sampler->next_2d();
        Point2f duv = (fp_sample[0] - 0.5f)*si.duv_dx + (fp_sample[1] - 0.5f)*si.duv_dy;
        SurfaceInteraction3f si_offset(si);
        si_offset.uv += duv;

        // Sample BSDF direction at that offset
        BSDFContext ctx;
        const BSDFPtr bsdf = si.bsdf();
        auto [bs, bsdf_weight] = bsdf->sample(ctx, si_offset, sampler->next_1d(), sampler->next_2d());
        bsdf_weight = si_offset.to_world_mueller(bsdf_weight, -bs.wo, si_offset.wi);    // No-op in unpolarized modes

        // Need to actually hit an emitter with that direction
        Vector3f wo = si_offset.to_world(bs.wo);
        auto [emitter, emitter_value, si_bsdf] = emitter_bsdf_sampling_lookahead(si_offset, wo, scene, sampler);
        if (emitter) {
            /* Compute pdf of what we just did, but again averaged over the pixel
               footprint using the LEAN approximation. */
            Float p_bsdf = mf_glints.lean_pdf(si_offset, bs.wo);
            /* Apply a constant offset to make sure the density is non-zero for all
               parts of the integrand. */
            p_bsdf += 1.f;

            /* Compute hypothetical pdf of the SMS strategy, again approximated with
               the usual direct illumination / emitter sampling pdf. */
            DirectionSample3f ds(si_bsdf, si_offset);
            ds.object = emitter;
            Float p_sms = scene->pdf_emitter_direction(si_offset, ds);

            Float mis = mis_weight(p_bsdf, p_sms);
            result += mis * bsdf_weight * emitter_value;
        }

        return result;
    }

    std::pair<bool, Spectrum>
    emitter_visibility_test(const SurfaceInteraction3f &si,
                            const EmitterInteraction &ei,
                            const Scene *scene,
                            ref<Sampler> sampler) const {
        bool footprint_visible = true;
        Spectrum light_throughput(1.f);

        if (ei.is_directional()) {
            // To support glints behind windows specifically: trace through dielectrics to envmap

            Vector3f dy = normalize(ei.p - si.p);
            Ray3f probe_ray(si.p, dy, si.time, si.wavelengths);
            SurfaceInteraction3f probe_si = scene->ray_intersect(probe_ray);

            while (true) {
                if (!probe_si.is_valid()) {
                    // Reached environment!
                    break;
                }
                if (!has_flag(probe_si.bsdf()->flags(), BSDFFlags::DeltaTransmission)) {
                    // A solid obstacle was found
                    footprint_visible = false;
                    break;
                }

                // Perform refraction
                BSDFContext ctx; ctx.type_mask = +BSDFFlags::DeltaTransmission;
                auto [bs, bsdf_weight] = probe_si.bsdf()->sample(ctx, probe_si, sampler->next_1d(), sampler->next_2d());
                light_throughput *= bsdf_weight;

                // Continue on ray and intersect next thing
                probe_ray = probe_si.spawn_ray(probe_si.to_world(bs.wo));
                probe_si = scene->ray_intersect(probe_ray);
            }
        } else {
            // Usual case
            Vector3f dy = si.p - ei.p;
            Float dist_y = norm(dy);
            dy *= rcp(dist_y);
            Ray3f ray_vis(ei.p, dy,
                        math::RayEpsilon<Float> * (1.f + hmax(abs(ei.p))),
                        dist_y * (1.f - math::RayEpsilon<Float>),
                        si.time, si.wavelengths);
            footprint_visible = !scene->ray_test(ray_vis);
        }

        return std::make_pair(footprint_visible, light_throughput);
    }

    std::tuple<const EmitterPtr, Spectrum, SurfaceInteraction3f>
    emitter_bsdf_sampling_lookahead(const SurfaceInteraction3f &si_offset,
                                    const Vector3f &wo,
                                    const Scene *scene,
                                    ref<Sampler> sampler) const {

        Spectrum light_throughput(1.f);

        Ray3f ray_bsdf = si_offset.spawn_ray(wo);
        SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray_bsdf);

        while (true) {
            const Emitter *emitter = si_bsdf.emitter(scene);
            if (emitter) {
                // Either we hit area light or escape into the environment map
                return std::make_tuple(emitter, light_throughput * emitter->eval(si_bsdf), si_bsdf);
            }

            if (!has_flag(si_bsdf.bsdf()->flags(), BSDFFlags::DeltaTransmission)) {
                // A solid obstacle was found
                return std::make_tuple(nullptr, Spectrum(0.f), si_bsdf);
            }

            // Perform refraction
            BSDFContext ctx; ctx.type_mask = +BSDFFlags::DeltaTransmission;
            auto [bs, bsdf_weight] = si_bsdf.bsdf()->sample(ctx, si_bsdf, sampler->next_1d(), sampler->next_2d());
            light_throughput *= bsdf_weight;

            // Continue on ray and intersect next thing
            ray_bsdf = si_bsdf.spawn_ray(si_bsdf.to_world(bs.wo));
            si_bsdf = scene->ray_intersect(ray_bsdf);
        }
    }

    std::string to_string() const override {
        return tfm::format("TeaserSMSPathIntegrator[\n"
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
    SMSConfig m_sms_config_caustics;
    SMSConfig m_sms_config_glints;

    bool m_glints_enabled;
    bool m_caustics_enabled;
    bool m_non_sms_paths_enabled;  // Optionally disable all non-SMS paths for Fig. 13
};

MTS_IMPLEMENT_CLASS_VARIANT(TeaserSMSPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(TeaserSMSPathIntegrator, "Teaser image SMS Path Tracer integrator");
NAMESPACE_END(mitsuba)

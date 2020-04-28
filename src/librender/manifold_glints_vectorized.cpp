#include <mitsuba/render/manifold_glints_vectorized.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/lean.h>
#include <iomanip>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT std::atomic<int> SpecularManifoldGlintsVectorized<Float, Spectrum>::stats_solver_failed(0);
MTS_VARIANT std::atomic<int> SpecularManifoldGlintsVectorized<Float, Spectrum>::stats_solver_succeeded(0);

MTS_VARIANT void
SpecularManifoldGlintsVectorized<Float, Spectrum>::init_sampler(ref<Sampler> sampler) {
    using UInt64P = uint64_array_t<FloatP>;
    if (!m_rng) {
        m_rng = std::make_unique<PCG32>();
        uint64_t seed_value = reinterpret_array<uint64_t>(sampler->next_1d());    // Maybe not the best way to seed...
        m_rng->seed(seed_value, PCG32_DEFAULT_STREAM + arange<UInt64P>());
    }
}

MTS_VARIANT void
SpecularManifoldGlintsVectorized<Float, Spectrum>::init(const Scene *scene,
                                                        const SMSConfig &config) {
    m_scene = scene;
    m_config = config;
    if (!m_config.biased) {
        Log(Warn, "SpecularManifoldGlintsVectorized: only supports the biased SMS variant!");
    }
}

MTS_VARIANT Spectrum
SpecularManifoldGlintsVectorized<Float, Spectrum>::specular_manifold_sampling(const Point3f &sensor_position,
                                                                    const SurfaceInteraction3f &si,
                                                                    ref<Sampler> sampler) const {
    ScopedPhase scope_phase(ProfilerPhase::SMSGlints);

    if (unlikely(!si.is_valid() || !si.shape->is_glinty())) {
        return 0.f;
    }

    Spectrum result(0.f);

    // Sample emitter interaction
    EmitterInteraction ei = SpecularManifold::sample_emitter_interaction(si, m_scene->emitters(), sampler);

    // Check visibility to emitter already here, otherwise skip to BSDF sampling strategy
    Vector3f emitter_direction = ei.p - si.p;
    Float emitter_dist = norm(emitter_direction);
    emitter_direction *= rcp(emitter_dist);
    Ray3f ray_emitter(si.p, emitter_direction,
                      math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                      emitter_dist * (1.f - math::RayEpsilon<Float>),
                      si.time, si.wavelengths);
    if (!m_scene->ray_test(ray_emitter)) {

        const ShapePtr shape = si.shape;
        Float alpha = shape->bsdf()->roughness();
        if (alpha > 0.f) {
            Log(Warn, "SpecularManifoldGlintsVectorized: only supports purely specular glints without roughness!");
        }

        // ----------------------- SMS Sampling strategy --------------------------

        bool sms_success = false;
        Spectrum sms_weight(0.f);

        // Biased SMS

        std::vector<Point2f> solutions;

        int num_samples = 0;
        while (num_samples < m_config.max_trials) {
            // Sample vector packet of glints at a time
            auto [success, res_uv] = sample_glint(sensor_position, ei, si);
            if (none(success)) {
                stats_solver_failed += VECTOR_PACKET_SIZE;
                num_samples += VECTOR_PACKET_SIZE;
                continue;
            }

            // Clustering takes place entry by entry
            for (size_t p = 0; p < VECTOR_PACKET_SIZE && num_samples < m_config.max_trials; ++p) {
                num_samples++;

                if (slice(success, p)) {
                    Point2f res_uv_p = slice(res_uv, p);
                    stats_solver_succeeded++;

                    bool duplicate = false;
                    for (size_t k = 0; k < solutions.size(); ++k) {
                        if (norm(res_uv_p - solutions[k]) < m_config.uniqueness_threshold) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate) {
                        continue;
                    }
                    solutions.push_back(res_uv_p);
                }
            }
        }

        /* Work through unique solutions that we found in batches.
           The full VECTOR_PACKET_SIZE vectors are not very efficiently utilized here unfortunately if
           few solutions are found. */
        size_t batches = (solutions.size() / VECTOR_PACKET_SIZE) + 1;
        for (size_t b = 0; b < batches; ++b) {
            UInt32P idx = (b+1)*arange<UInt32P>();

            MaskP active = idx < solutions.size();
            Point2fP uv = gather<Point2fP, 0, true>(solutions.data(), idx, active);

            sms_weight += hsum(evaluate_glint_contribution(sensor_position, ei, si, uv, active));
        }

        sms_success = solutions.size() > 0;

        if (sms_success) {
            // Evaluate the complate path throughput
            SensorPtr sensor = m_scene->sensors()[0];
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
            if (!ei.is_delta()) {
                DirectionSample3f ds;
                ds.object = ei.emitter;
                ds.p = ei.p; ds.n = ei.n; ds.d = ei.d;
                ds.dist = emitter_dist;
                p_sms = m_scene->pdf_emitter_direction(si, ds);
            }

            /* Compute hypothetical pdf for this direction using BSDF sampling for
               the full directional density of the BSDF averaged over the footprint.
               This is tricky to evaluate exactly, but we can compute a good
               approximation based on LEAN mapping. */
            Float p_bsdf = lean_pdf(si, si.to_local(emitter_direction));
            /* Apply a constant offset to make sure the density is non-zero for all
               parts of the integrand. */
            p_bsdf += 1.f;

            Float mis = mis_weight(p_sms, p_bsdf);
            if (m_config.bsdf_strategy_only) mis = 0.f;

            result += mis * We * sms_weight * ei.weight;
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

    Ray3f ray_bsdf = si_offset.spawn_ray(si_offset.to_world(bs.wo));
    SurfaceInteraction3f si_bsdf = m_scene->ray_intersect(ray_bsdf);
    const EmitterPtr emitter = si_bsdf.emitter(m_scene);

    // We we hit an emitter with it?
    if (emitter) {
        /* Compute pdf of what we just did, but again averaged over the pixel
           footprint using the LEAN approximation. */
        Float p_bsdf = lean_pdf(si_offset, bs.wo);
        /* Apply a constant offset to make sure the density is non-zero for all
           parts of the integrand. */
        p_bsdf += 1.f;

        /* Compute hypothetical pdf of the SMS strategy, again approximated with
           the usual direct illumination / emitter sampling pdf. */
        DirectionSample3f ds(si_bsdf, si_offset);
        ds.object = emitter;
        Float p_sms = m_scene->pdf_emitter_direction(si_offset, ds);

        Float mis = mis_weight(p_bsdf, p_sms);
        if (m_config.sms_strategy_only) mis = 0.f;

        result += mis * bsdf_weight * emitter->eval(si_bsdf);
    }

    return result;
}

MTS_VARIANT std::pair<typename SpecularManifoldGlintsVectorized<Float, Spectrum>::MaskP,
                      typename SpecularManifoldGlintsVectorized<Float, Spectrum>::Point2fP>
SpecularManifoldGlintsVectorized<Float, Spectrum>::sample_glint(const Point3f &sensor_position,
                                                                const EmitterInteraction &ei,
                                                                const SurfaceInteraction3f &si) const {

    // Check sides
    Vector3f wi = si.to_local(normalize(sensor_position - si.p)),
             wo = si.to_local(normalize(ei.p - si.p));
    if (Frame3f::cos_theta(wi) <= 0.f || Frame3f::cos_theta(wo) <= 0.f) {
        return { false, 0.f };
    }

    // Compute current (fixed!) half-vector in this footprint
    Vector3f h = normalize(wi + wo);

    // Put into slope space
    Point2f target_slope(-h[0]/h[2], -h[1]/h[2]);

    // We now need to find a UV position where the slope matches the target slope..

    // Sample uniformly random position in footprint
    FloatP fp_sample_u = m_rng->next_float32(),
           fp_sample_v = m_rng->next_float32();
    Point2fP fp_sample(fp_sample_u, fp_sample_v);
    Point2fP duv = (fp_sample[0] - 0.5f)*Vector2fP(si.duv_dx) +
                   (fp_sample[1] - 0.5f)*Vector2fP(si.duv_dy);
    Point2fP uv_init = si.uv + duv;

    // Run the newton solver to find a valid position
    auto [success, uv_final] = newton_solver(target_slope, uv_init, si);

    return std::pair(success, uv_final);
}

MTS_VARIANT typename SpecularManifoldGlintsVectorized<Float, Spectrum>::SpectrumP
SpecularManifoldGlintsVectorized<Float, Spectrum>::evaluate_glint_contribution(const Point3f &sensor_position,
                                                                               const EmitterInteraction &ei,
                                                                               const SurfaceInteraction3f &si,
                                                                               const Point2fP &uv,
                                                                               MaskP active) const {
    SpectrumP path_throughput(1.f);
    path_throughput *= specular_reflectance(si, normalize(ei.p - si.p), uv, active);

    // Set up emitter vertex
    auto [success_e, vy] = SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, si.p, si.time, si.wavelengths);
    if (!success_e) {
        return 0.f;
    }

    // Set up camera vertex
    ManifoldVertex vx(sensor_position);
    Vector3f dx = normalize(si.p - sensor_position);
    vx.n = vx.gn = dx;
    std::tie(vx.dp_du, vx.dp_dv) = coordinate_system(vx.n);

/* Compared to a fully vectorized version of the geometric term evaluation below,
   we found that it is often times more efficient to have a small scalar loop
   over the found solutions here. In most cases, we only find a single (or very
   few solutions), so the added overhead from the vectorization is not paying off. */
#if 1
    FloatP G(0.f);
    for (size_t p = 0; p < VECTOR_PACKET_SIZE; ++p) {
        if (active[p]) {
            SurfaceInteraction3f si_offset(si);
            si_offset.uv = slice(uv, p);
            ManifoldVertex vtx(si_offset, 0.f);
            vtx.make_orthonormal();
            slice(G, p) = geometric_term(vx, vtx, vy, true);
        }
    }
    path_throughput *= G;
#else
    if (count(active) == 1) {
        // Fast lane for single found solution
        SurfaceInteraction3f si_offset(si);
        si_offset.uv = slice(uv, 0);
        ManifoldVertex vtx(si_offset, 0.f);
        vtx.make_orthonormal();

        path_throughput *= geometric_term(vx, vtx, vy, true);
    } else {
        /* Set up packet ManifoldVertex with the right UV positions found by Newton
           solver. Only fill essential fields required for G computation below.
           Follows NormalmapBSDF::frame/frame_derivative */
        ManifoldVertexP vtx;
        vtx.p = si.p; vtx.dp_du = si.dp_du; vtx.dp_dv = si.dp_dv; vtx.gn = si.n;

        Vector3fP n, dn_du, dn_dv;
        const BSDFPtr bsdf = si.shape->bsdf();
        Point2fP s = slope(bsdf, uv, active);
        n = Vector3fP(-s[0], -s[1], 1.f);
        FloatP inv_norm = rcp(norm(n));
        n *= inv_norm;
        auto [ds_du, ds_dv] = slope_derivative(bsdf, uv, active);
        dn_du = Vector3fP(ds_du[0], ds_du[1], 1.f);
        dn_dv = Vector3fP(ds_dv[0], ds_dv[1], 1.f);
        dn_du = inv_norm * (dn_du - n*dot(n, dn_du));
        dn_dv = inv_norm * (dn_dv - n*dot(n, dn_dv));

        Frame3f base = compute_shading_frame(si.sh_frame.n, si.dp_du);
        auto [si_dn_du, si_dn_dv] = si.shape->normal_derivative(si);
        auto [dbase_du, dbase_dv] = compute_shading_frame_derivative(si.sh_frame.n, si.dp_du, si_dn_du, si_dn_dv);

        vtx.n = Vector3fP(base.s) * n[0] + Vector3fP(base.t) * n[1] + Vector3fP(base.n) * n[2];
        FloatP inv_length_n = rcp(norm(vtx.n));
        vtx.n *= inv_length_n;

        Vector3fP tmp_u = Vector3fP(base.s) * dn_du[0] + Vector3fP(base.t) * dn_du[1] + Vector3fP(base.n) * dn_du[2] +
                          Vector3fP(dbase_du.s) * n[0] + Vector3fP(dbase_du.t) * n[1] + Vector3fP(dbase_du.n) * n[2];
        Vector3fP tmp_v = Vector3fP(base.s) * dn_dv[0] + Vector3fP(base.t) * dn_dv[1] + Vector3fP(base.n) * dn_dv[2] +
                          Vector3fP(dbase_dv.s) * n[0] + Vector3fP(dbase_dv.t) * n[1] + Vector3fP(dbase_dv.n) * n[2];

        vtx.dn_du = inv_length_n * tmp_u,
        vtx.dn_dv = inv_length_n * tmp_v;
        vtx.dn_du -= vtx.n * dot(vtx.dn_du, vtx.n);
        vtx.dn_dv -= vtx.n * dot(vtx.dn_dv, vtx.n);
        vtx.make_orthonormal();

        path_throughput *= geometric_term(vx, vtx, vy, active);
    }
#endif

    return select(active, path_throughput, 0.f);
}

MTS_VARIANT std::pair<typename SpecularManifoldGlintsVectorized<Float, Spectrum>::MaskP,
                      typename SpecularManifoldGlintsVectorized<Float, Spectrum>::Point2fP>
SpecularManifoldGlintsVectorized<Float, Spectrum>::newton_solver(const Point2f &target_slope,
                                                                 const Point2fP &uv_init,
                                                                 const SurfaceInteraction3f &si) const {
    MaskP active = true,
          success = false;
    Point2fP uv = uv_init;

    // Approximate footprint as parallelogram and compute its corners
    Point2f fp_a = si.uv + 0.5f*si.duv_dx + 0.5f*si.duv_dy,
            fp_b = si.uv + 0.5f*si.duv_dx - 0.5f*si.duv_dy,
            fp_c = si.uv - 0.5f*si.duv_dx - 0.5f*si.duv_dy,
            fp_d = si.uv - 0.5f*si.duv_dx + 0.5f*si.duv_dy;

    size_t iterations = 0;
    FloatP beta = 1.f;
    while (iterations < m_config.max_iterations && any(active)) {
        // Evaluate constraint function value and compute step
        auto [step_success, C, step] = compute_step(target_slope, uv, si, active);

        // Check if some vector lanes arrived at target
        MaskP arrived = (step_success & norm(C) < m_config.solver_threshold);
        masked(success, arrived) = true;
        // Disable lanes that succeeded, no work is required here anymore
        masked(active, success) = false;

        // Potentially terminate if all lanes are done
        if (none(active)) {
            break;
        }

        // Make proposal
        Point2fP uv_prop = uv + m_config.step_scale*beta * step;

        // Did we step outside the footprint?
        MaskP inside = inside_parallelogram(uv_prop, fp_a, fp_b, fp_c, fp_d);

        // Potentially decrease / increase step size
        masked(beta, !inside & step_success & active) *= 0.5f;
        masked(beta,  inside & step_success & active) = min(1.f, 2.f*beta);

        // Accept step
        masked(uv, inside & step_success & active) = uv_prop;

        iterations++;
    }

    return { success, uv };
}

MTS_VARIANT std::tuple<typename SpecularManifoldGlintsVectorized<Float, Spectrum>::MaskP,
                       typename SpecularManifoldGlintsVectorized<Float, Spectrum>::Point2fP,
                       typename SpecularManifoldGlintsVectorized<Float, Spectrum>::Vector2fP>
SpecularManifoldGlintsVectorized<Float, Spectrum>::compute_step(const Point2f &target_slope,
                                                                const Point2fP &uv,
                                                                const SurfaceInteraction3f &si,
                                                                MaskP active) const {
    const BSDFPtr bsdf = si.bsdf();

    // Evaluate constraint
    Point2fP current_slope = slope(bsdf, uv, active);
    Point2fP dS = current_slope - target_slope;

    auto [dslope_du, dslope_dv] = slope_derivative(bsdf, uv, active);

    Matrix2fP dS_dX(0.f);
    dS_dX(0,0) = dslope_du[0];
    dS_dX(1,0) = dslope_du[1];
    dS_dX(0,1) = dslope_dv[0];
    dS_dX(1,1) = dslope_dv[1];

    FloatP determinant = det(dS_dX);
    MaskP singular = abs(determinant) < 1e-6f;
    Matrix2fP dX_dS = inverse(dS_dX);

    Vector2fP dX = dX_dS * dS;
    return { !singular & active, dS, dX };
}

MTS_VARIANT typename SpecularManifoldGlintsVectorized<Float, Spectrum>::SpectrumP
SpecularManifoldGlintsVectorized<Float, Spectrum>::specular_reflectance(const SurfaceInteraction3f &si,
                                                                        const Vector3f &wo,
                                                                        const Point2fP &uv,
                                                                        MaskP active) const {
    if (!si.is_valid()) return 0.f;
    const BSDFPtr bsdf = si.shape->bsdf();
    if (!bsdf) return 0.f;

    SpectrumP bsdf_val(1.f);

    // Do same conversion as in NormalmapBSDF::frame (but with a packet of uv)
    Point2fP s = slope(bsdf, uv, active);
    Vector3fP n_local = Vector3fP(-s[0], -s[1], 1.f);
    Frame3f frame = compute_shading_frame(si.sh_frame.n, si.dp_du);
    Vector3fP n_world = Vector3fP(frame.s) * n_local[0] +
                        Vector3fP(frame.t) * n_local[1] +
                        Vector3fP(frame.n) * n_local[2];
    n_world = normalize(n_world);
    FloatP cos_theta = dot(n_world, wo);

    Complex<Spectrum> ior = bsdf->ior(si);  // Assume IOR stays constant within pixel footprint.
    if (all(eq(imag(ior), 0.f))) {
        auto [F, cos_theta_t, eta_it, eta_ti] = fresnel(SpectrumP(abs(cos_theta)), SpectrumP(real(ior)));
        bsdf_val = F;
    } else {
        bsdf_val = fresnel_conductor(SpectrumP(abs(cos_theta)), Complex<SpectrumP>(ior));
    }
    return select(active, bsdf_val, 0.f);
}

MTS_VARIANT typename SpecularManifoldGlintsVectorized<Float, Spectrum>::Point2fP
SpecularManifoldGlintsVectorized<Float, Spectrum>::slope(const BSDFPtr bsdf,
                                                         const Point2fP &uv_,
                                                         MaskP active) const {
    using Vector5fP = Vector<FloatP, 5>;

    auto [data, res, tiles] = bsdf->lean_pointer();
    Float duv = rcp(Float(res));

    Point2fP uv = uv_;
    uv *= tiles;
    uv -= floor(uv);

    Point2fP pos = (uv - 0.5f*duv)*res;
             pos = clamp(pos, Point2f(0), Point2f(res-1));
    Point2uP p_min = Point2uP(floor(pos)),
             p_max = p_min + 1;
    p_min = clamp(p_min, 0, res-1);
    p_max = clamp(p_max, 0, res-1);

    UInt32P idx00 = p_min.y()*res + p_min.x(),
            idx10 = p_min.y()*res + p_max.x(),
            idx01 = p_max.y()*res + p_min.x(),
            idx11 = p_max.y()*res + p_max.x();

    Point2fP w1 = pos - p_min,
             w0 = 1.f - w1;

    Vector5fP v00 = gather<Vector5fP, 0, true>(data, idx00, active),
              v10 = gather<Vector5fP, 0, true>(data, idx10, active),
              v01 = gather<Vector5fP, 0, true>(data, idx01, active),
              v11 = gather<Vector5fP, 0, true>(data, idx11, active);
    Vector5fP v0 = fmadd(w0.x(), v00, w1.x() * v10),
              v1 = fmadd(w0.x(), v01, w1.x() * v11);
    Vector5fP v = fmadd(w0.y(), v0, w1.y() * v1);

    Point2fP slope(v[0], v[1]);
    return slope;
}

MTS_VARIANT std::pair<typename SpecularManifoldGlintsVectorized<Float, Spectrum>::Vector2fP,
                      typename SpecularManifoldGlintsVectorized<Float, Spectrum>::Vector2fP>
SpecularManifoldGlintsVectorized<Float, Spectrum>::slope_derivative(const BSDFPtr bsdf,
                                                                    const Point2fP &uv_,
                                                                    MaskP active) const {
    using Vector5fP = Vector<FloatP, 5>;

    auto [data, res, tiles] = bsdf->lean_pointer();
    Float duv = rcp(Float(res));

    Point2fP uv = uv_;
    uv *= tiles;
    uv -= floor(uv);

    Point2fP pos = (uv - 0.5f*duv)*res;
             pos = clamp(pos, Point2f(0), Point2f(res-1));
    Point2uP p_min = Point2uP(floor(pos)),
             p_max = p_min + 1;
    p_min = clamp(p_min, 0, res-1);
    p_max = clamp(p_max, 0, res-1);

    UInt32P idx00 = p_min.y()*res + p_min.x(),
            idx10 = p_min.y()*res + p_max.x(),
            idx01 = p_max.y()*res + p_min.x(),
            idx11 = p_max.y()*res + p_max.x();

    Point2fP w = pos - p_min;

    Vector5fP v00 = gather<Vector5fP, 0, true>(data, idx00, active),
              v10 = gather<Vector5fP, 0, true>(data, idx10, active),
              v01 = gather<Vector5fP, 0, true>(data, idx01, active),
              v11 = gather<Vector5fP, 0, true>(data, idx11, active);
    Vector5fP tmp = v01 + v10 - v11;
    Vector5fP tmp_u = (v10 + v00*(w[1] - 1.f) - tmp*w[1]) * Float(res),
              tmp_v = (v01 + v00*(w[0] - 1.f) - tmp*w[0]) * Float(res);

    Vector2fP dslope_du(-tmp_u[0], -tmp_u[1]),
              dslope_dv(-tmp_v[0], -tmp_v[1]);

    dslope_du *= tiles;
    dslope_dv *= tiles;

    return { dslope_du, dslope_dv };
}

MTS_VARIANT Float
SpecularManifoldGlintsVectorized<Float, Spectrum>::lean_pdf(const SurfaceInteraction3f &si,
                                                  const Vector3f &wo) const {
    // Use a local LEAN mapping approximation for the PDF used to MIS the glints
    Float cos_theta_i = Frame3f::cos_theta(si.wi),
          cos_theta_o = Frame3f::cos_theta(wo);
    if (cos_theta_i < 0.f || cos_theta_o < 0.f) {
        return 0.f;
    }

    const BSDF *bsdf = si.bsdf();
    Float roughness = bsdf->roughness();

    auto [mu_lean, sigma_lean] = bsdf->lean(si);
    LEANParameters params = LEAN<Float, Spectrum>::p_from_lean_and_base(mu_lean, sigma_lean, roughness, roughness);

    Vector3f H = normalize(wo + si.wi);
    Float G = LEAN<Float, Spectrum>::gaf(H, wo, si.wi, params);
    if (G > 0) {
        return LEAN<Float, Spectrum>::vndf(H, si.wi, params) / (4.f*dot(si.wi, H));
    } else {
        return 0.f;
    }
}

MTS_VARIANT void SpecularManifoldGlintsVectorized<Float, Spectrum>::print_statistics() {
    Float solver_success_ratio = Float(stats_solver_succeeded) / (stats_solver_succeeded + stats_solver_failed),
          solver_fail_ratio    = Float(stats_solver_failed)    / (stats_solver_succeeded + stats_solver_failed);

    std::cout << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "    Specular Manifold Sampling Statistics" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << std::setw(25) << std::left << "walks succeeded: "
              << std::setw(10) << std::right << stats_solver_succeeded << " "
              << std::setw(8) << "(" << 100*solver_success_ratio << "%)" << std::endl;
    std::cout << std::setw(25) << std::left << "walks failed: "
              << std::setw(10) << std::right << stats_solver_failed << " "
              << std::setw(8) << "(" << 100*solver_fail_ratio << "%)" << std::endl;
    std::cout << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << std::endl;
}

MTS_INSTANTIATE_CLASS(SpecularManifoldGlintsVectorized)
NAMESPACE_END(mitsuba)
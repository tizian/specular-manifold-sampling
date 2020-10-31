#include <mitsuba/render/manifold_glints.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/lean.h>
#include <iomanip>

NAMESPACE_BEGIN(mitsuba)

template<typename T>
inline void update_max(std::atomic<T> & atom, const T val) {
  for(T atom_val=atom; atom_val < val && !atom.compare_exchange_weak(atom_val, val, std::memory_order_relaxed););
}

MTS_VARIANT std::atomic<int> SpecularManifoldGlints<Float, Spectrum>::stats_solver_failed(0);
MTS_VARIANT std::atomic<int> SpecularManifoldGlints<Float, Spectrum>::stats_solver_succeeded(0);
MTS_VARIANT std::atomic<int> SpecularManifoldGlints<Float, Spectrum>::stats_bernoulli_trial_calls(0);
MTS_VARIANT std::atomic<int> SpecularManifoldGlints<Float, Spectrum>::stats_bernoulli_trial_iterations(0);
MTS_VARIANT std::atomic<int> SpecularManifoldGlints<Float, Spectrum>::stats_bernoulli_trial_iterations_max(0);

MTS_VARIANT void
SpecularManifoldGlints<Float, Spectrum>::init(const Scene *scene,
                                              const SMSConfig &config) {
    m_scene = scene;
    m_config = config;
}

MTS_VARIANT Spectrum
SpecularManifoldGlints<Float, Spectrum>::specular_manifold_sampling(const Point3f &sensor_position,
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

        // Sample offset normal at specular surface
        Vector3f n_offset(0.f, 0.f, 1.f);
        Float p_offset = 1.f;
        Float alpha = shape->bsdf()->roughness();
        if (alpha > 0.f) {
            // Assume isotropic Beckmann distribution for the surface roughness.
            MicrofacetDistribution<Float, Spectrum> distr(MicrofacetType::Beckmann, alpha, alpha, true);
            Vector3f wi = si.to_local(normalize(sensor_position - si.p));
            std::tie(n_offset, p_offset) = distr.sample(wi, sampler->next_2d());
        }

        // ----------------------- SMS Sampling strategy --------------------------

        bool sms_success = false;
        Spectrum sms_weight(0.f);
        if (!m_config.biased) {
            // Unbiased SMS

            auto [success, uv_final, unused] = sample_glint(sensor_position, ei, si, sampler, n_offset);
            if (success) {
                stats_solver_succeeded++;
                sms_success = true;

                // We sampled a valid glint position, now compute its contribution
                Spectrum specular_val = evaluate_glint_contribution(sensor_position, ei, si, uv_final);

                // Now estimate the (inverse) probability of this whole process with Bernoulli trials
                Float inv_prob_estimate = 1.f;
                int iterations = 1;
                stats_bernoulli_trial_calls++;
                while (true) {
                    ScopedPhase scope_phase(ProfilerPhase::SMSGlintsBernoulliTrials);

                    auto [sucess_trial, uv_final_trial, unused_trial] = sample_glint(sensor_position, ei, si, sampler, n_offset);
                    if (sucess_trial && norm(uv_final_trial - uv_final) < m_config.uniqueness_threshold) {
                        break;
                    }

                    inv_prob_estimate += 1.f;
                    iterations++;

                    if (m_config.max_trials > 0 && iterations > m_config.max_trials) {
                        /* There is a tiny chance always to sample super weird positions
                           that will never occur again due to small numerical imprecisions.
                           So setting a (super conservative) threshold here can help
                           to avoid infinite loops. */
                        inv_prob_estimate = 0.f;
                        break;
                    }
                }
                stats_bernoulli_trial_iterations += iterations;
                update_max(stats_bernoulli_trial_iterations_max, iterations);

                sms_weight = specular_val * inv_prob_estimate;
            } else {
                stats_solver_failed++;
            }
        } else {
            // Biased SMS

            std::vector<Point2f> solutions;
            for (int m = 0; m < m_config.max_trials; ++m) {
                auto [success, uv_final, unused] = sample_glint(sensor_position, ei, si, sampler, n_offset);
                if (!success) {
                    stats_solver_failed++;
                    continue;
                }
                stats_solver_succeeded++;

                // Check if this is a new and unique solution
                bool duplicate = false;
                for (size_t k = 0; k < solutions.size(); ++k) {
                    if (norm(uv_final - solutions[k]) < m_config.uniqueness_threshold) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) {
                    continue;
                }
                solutions.push_back(uv_final);

                sms_weight += evaluate_glint_contribution(sensor_position, ei, si, uv_final);
            }
            sms_success = solutions.size() > 0;
        }

        sms_weight /= p_offset;

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

MTS_VARIANT std::tuple<typename SpecularManifoldGlints<Float, Spectrum>::Mask,
                       typename SpecularManifoldGlints<Float, Spectrum>::Point2f,
                       typename SpecularManifoldGlints<Float, Spectrum>::Point2f>
SpecularManifoldGlints<Float, Spectrum>::sample_glint(const Point3f &sensor_position,
                                                      const EmitterInteraction &ei,
                                                      const SurfaceInteraction3f &si,
                                                      ref<Sampler> sampler,
                                                      const Vector3f &n_offset,
                                                      const Point2f xi_start) const {
    // Check sides
    Vector3f wi = si.to_local(normalize(sensor_position - si.p)),
             wo = si.to_local(normalize(ei.p - si.p));
    if (Frame3f::cos_theta(wi) <= 0.f || Frame3f::cos_theta(wo) <= 0.f) {
        return std::make_tuple(false, 0.f, 0.f);
    }

    // Compute current (fixed!) half-vector in this footprint
    Vector3f h = normalize(wi + wo);

    // Put into slope space and already combine with (potential) offset normal
    Point2f h_s(-h[0]/h[2], -h[1]/h[2]),
            o_s(-n_offset[0]/n_offset[2], -n_offset[1]/n_offset[2]);
    Point2f target_slope = h_s + o_s;

    // We now need to find a UV position where the slope matches the target slope..

    // Sample uniformly random position in footprint
    Point2f fp_sample = sampler->next_2d();
    if (any(neq(xi_start, 1.f))) {
        fp_sample = xi_start;
    }

    Point2f duv = (fp_sample[0] - 0.5f)*si.duv_dx +
                  (fp_sample[1] - 0.5f)*si.duv_dy;
    Point2f uv_init = si.uv + duv;

    // Run the newton solver to find a valid position
    auto [success, uv_final] = newton_solver(target_slope, uv_init, si);

    return std::make_tuple(success, uv_final, uv_init);
}

MTS_VARIANT Spectrum
SpecularManifoldGlints<Float, Spectrum>::evaluate_glint_contribution(const Point3f &sensor_position,
                                                                     const EmitterInteraction &ei,
                                                                     const SurfaceInteraction3f &si,
                                                                     const Point2f &uv) const {
    // Emitter vertex
    auto [success_e, vy] = SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, si.p, si.time, si.wavelengths);
    if (!success_e) {
        return 0.f;
    }

    // Camera vertex
    ManifoldVertex vx(sensor_position);
    Vector3f dx = normalize(si.p - sensor_position);
    vx.n = vx.gn = dx;
    std::tie(vx.dp_du, vx.dp_dv) = coordinate_system(vx.n);

    // ManifoldVertex with the right UV position found by the Newton solver
    SurfaceInteraction3f si_offset(si);
    si_offset.uv = uv;
    ManifoldVertex vtx(si_offset, 0.f);
    vtx.make_orthonormal();

    Spectrum path_throughput(1.f);
    path_throughput *= specular_reflectance(si_offset, normalize(ei.p - si_offset.p));
    path_throughput *= geometric_term(vx, vtx, vy);
    return path_throughput;
}

MTS_VARIANT std::pair<typename SpecularManifoldGlints<Float, Spectrum>::Mask,
                      typename SpecularManifoldGlints<Float, Spectrum>::Point2f>
SpecularManifoldGlints<Float, Spectrum>::newton_solver(const Point2f &target_slope,
                                                       const Point2f &uv_init,
                                                       const SurfaceInteraction3f &si) const {
    Point2f uv = uv_init;

    // Approximate footprint as parallelogram and compute its corners
    Point2f fp_a = si.uv + 0.5f*si.duv_dx + 0.5f*si.duv_dy,
            fp_b = si.uv + 0.5f*si.duv_dx - 0.5f*si.duv_dy,
            fp_c = si.uv - 0.5f*si.duv_dx - 0.5f*si.duv_dy,
            fp_d = si.uv - 0.5f*si.duv_dx + 0.5f*si.duv_dy;

    // Newton iterations
    bool success = false;
    size_t iterations = 0;
    Float beta = 1.f;
    while (iterations < m_config.max_iterations) {
        // Evaluate constraint function value and compute step
        auto [step_success, C, step] = compute_step(target_slope, uv, si);
        if (!step_success) {
            break;
        }

        // Check for success
        if (norm(C) < m_config.solver_threshold) {
            success = true;
            break;
        }

        // Make proposal
        Point2f uv_prop = uv - m_config.step_scale*beta * step;

        // Did we step outside the footprint?
        if (!inside_parallelogram(uv_prop, fp_a, fp_b, fp_c, fp_d)) {
            beta = 0.5f*beta;
            iterations++;
            continue;
        }

        beta = min(1.f, 2.f*beta);
        uv = uv_prop;

        iterations++;
    }

    return std::pair(success, uv);
}

MTS_VARIANT std::tuple<typename SpecularManifoldGlints<Float, Spectrum>::Mask,
                       typename SpecularManifoldGlints<Float, Spectrum>::Point2f,
                       typename SpecularManifoldGlints<Float, Spectrum>::Vector2f>
SpecularManifoldGlints<Float, Spectrum>::compute_step(const Point2f &target_slope,
                                                      const Point2f &uv,
                                                      const SurfaceInteraction3f &si) const {
    const BSDFPtr bsdf = si.bsdf();

    // Evaluate constraint
    Point2f current_slope = bsdf->slope(uv);
    Point2f dS = current_slope - target_slope;

    auto [dslope_du, dslope_dv] = bsdf->slope_derivative(uv);

    Matrix2f dS_dX(0.f);
    dS_dX(0,0) = dslope_du[0];
    dS_dX(1,0) = dslope_du[1];
    dS_dX(0,1) = dslope_dv[0];
    dS_dX(1,1) = dslope_dv[1];

    Float determinant = det(dS_dX);
    if (abs(determinant) < 1e-6f) {
        return std::make_tuple(false, 0.f, 0.f);
    }
    Matrix2f dX_dS = inverse(dS_dX);
    Vector2f dX = dX_dS * dS;

    return std::make_tuple(true, dS, dX);
}

MTS_VARIANT Spectrum
SpecularManifoldGlints<Float, Spectrum>::specular_reflectance(const SurfaceInteraction3f &si,
                                                              const Vector3f &wo) const {
    if (!si.is_valid()) return 0.f;
    const BSDFPtr bsdf = si.shape->bsdf();
    if (!bsdf) return 0.f;

    Spectrum bsdf_val(1.f);
    if (bsdf->roughness() > 0.f) {
        // Glossy BSDF: evaluate BSDF and transform to half-vector domain.
        BSDFContext ctx;
        Vector3f wo_l = si.to_local(wo);
        bsdf_val = bsdf->eval(ctx, si, wo_l);

        /* Compared to Eq. 6 in [Hanika et al. 2015 (MNEE)], two terms are omitted:
           1) abs_dot(wo, n) is part of BSDF::eval
           2) abs_dot(h, n)  is part of the Microfacet distr. (also in BSDF::eval) */
        Vector3f h_l = normalize(si.wi + wo_l);
        bsdf_val *= 4.f*abs_dot(wo_l, h_l);
    } else {
        // Delta BSDF: just account for Fresnel term
        Frame3f frame = bsdf->frame(si, 0.f);
        Float cos_theta = dot(frame.n, wo);

        Complex<Spectrum> ior = bsdf->ior(si);
        if (all(eq(imag(ior), 0.f))) {
            auto [F_, cos_theta_t, eta_it, eta_ti] = fresnel(Spectrum(abs(cos_theta)), real(ior));
            bsdf_val = F_;
        } else {
            bsdf_val = fresnel_conductor(Spectrum(abs(cos_theta)), ior);
        }
    }

    return bsdf_val;
}

MTS_VARIANT Float
SpecularManifoldGlints<Float, Spectrum>::geometric_term(const ManifoldVertex &v0,
                                                        const ManifoldVertex &v1,
                                                        const ManifoldVertex &v2) const {
    Vector3f wi = v0.p - v1.p;
    Float ili = norm(wi);
    if (ili < 1e-3f) {
        return 0.f;
    }
    ili = rcp(ili);
    wi *= ili;

    Vector3f wo = v2.p - v1.p;
    Float ilo = norm(wo);
    if (ilo < 1e-3f) {
        return 0.f;
    }
    ilo = rcp(ilo);
    wo *= ilo;

    Matrix2f dc1_dx0, dc2_dx1, dc2_dx2;
    if (v2.fixed_direction) {
        /* This case is actually a bit more tricky as we're now in a situation
           with two "specular" constraints. As a consequence, we need to solve
           a bigger matrix system, so we prepare a few additional terms. */

        // Derivative of directional light constraint w.r.t. v1
        Vector3f dc2_du1 = ilo * (v1.dp_du - wo * dot(wo, v1.dp_du)),
                 dc2_dv1 = ilo * (v1.dp_dv - wo * dot(wo, v1.dp_dv));
        dc2_dx1 = Matrix2f(
            dot(dc2_du1, v2.dp_du), dot(dc2_dv1, v2.dp_du),
            dot(dc2_du1, v2.dp_dv), dot(dc2_dv1, v2.dp_dv)
        );

        // Derivative of directional light constraint w.r.t. v2
        Vector3f dc2_du2 = -ilo * (v2.dp_du - wo * dot(wo, v2.dp_du)),
                 dc2_dv2 = -ilo * (v2.dp_dv - wo * dot(wo, v2.dp_dv));
        dc2_dx2 = Matrix2f(
            dot(dc2_du2, v2.dp_du), dot(dc2_dv2, v2.dp_du),
            dot(dc2_du2, v2.dp_dv), dot(dc2_dv2, v2.dp_dv)
        );
    }

    // Setup generalized half-vector
    Vector3f h = wi + wo;
    Float ilh = rcp(norm(h));
    h *= ilh;

    ilo *= ilh;
    ili *= ilh;

    // Local shading tangent frame
    Float dot_dpdu_n = dot(v1.dp_du, v1.n),
          dot_dpdv_n = dot(v1.dp_dv, v1.n);
    Vector3f s = v1.dp_du - dot_dpdu_n * v1.n,
             t = v1.dp_dv - dot_dpdv_n * v1.n;

    Vector3f dh_du, dh_dv;

    if (v2.fixed_direction) {
        // Derivative of specular constraint w.r.t. v0
        dh_du = ili * (v0.dp_du - wi * dot(wi, v0.dp_du));
        dh_dv = ili * (v0.dp_dv - wi * dot(wi, v0.dp_dv));
        dh_du -= h * dot(dh_du, h);
        dh_dv -= h * dot(dh_dv, h);
        dc1_dx0 = Matrix2f(
            dot(dh_du, s), dot(dh_dv, s),
            dot(dh_du, t), dot(dh_dv, t)
        );
    }

    // Derivative of specular constraint w.r.t. v1
    dh_du = -v1.dp_du * (ili + ilo) + wi * (dot(wi, v1.dp_du) * ili)
                                    + wo * (dot(wo, v1.dp_du) * ilo);
    dh_dv = -v1.dp_dv * (ili + ilo) + wi * (dot(wi, v1.dp_dv) * ili)
                                    + wo * (dot(wo, v1.dp_dv) * ilo);
    dh_du -= h * dot(dh_du, h);
    dh_dv -= h * dot(dh_dv, h);

    Float dot_h_n    = dot(h, v1.n),
          dot_h_dndu = dot(h, v1.dn_du),
          dot_h_dndv = dot(h, v1.dn_dv);
    Matrix2f dc1_dx1(
        dot(dh_du, s) - dot(v1.dp_du, v1.dn_du) * dot_h_n - dot_dpdu_n * dot_h_dndu,
        dot(dh_dv, s) - dot(v1.dp_du, v1.dn_dv) * dot_h_n - dot_dpdu_n * dot_h_dndv,
        dot(dh_du, t) - dot(v1.dp_dv, v1.dn_du) * dot_h_n - dot_dpdv_n * dot_h_dndu,
        dot(dh_dv, t) - dot(v1.dp_dv, v1.dn_dv) * dot_h_n - dot_dpdv_n * dot_h_dndv
    );

    // Derivative of specular constraint w.r.t. v2
    dh_du = ilo * (v2.dp_du - wo * dot(wo, v2.dp_du));
    dh_dv = ilo * (v2.dp_dv - wo * dot(wo, v2.dp_dv));
    dh_du -= h * dot(dh_du, h);
    dh_dv -= h * dot(dh_dv, h);
    Matrix2f dc1_dx2(
        dot(dh_du, s), dot(dh_dv, s),
        dot(dh_du, t), dot(dh_dv, t)
    );

    Float G = 0.f;
    if (v2.fixed_direction) {
        // Invert 2x2 block matrix system
        Float determinant = det(dc2_dx2);
        if (abs(determinant) < 1e-6f) {
            return 0.f;
        }
        Matrix2f Li = inverse(dc2_dx2);
        Matrix2f tmp = Li * dc2_dx1;
        Matrix2f m = dc1_dx1 - dc1_dx2 * tmp;
        determinant = det(m);
        if (abs(determinant) < 1e-6f) {
            return 0.f;
        }
        Li = inverse(m);
        Matrix2f sol1 = -Li * dc1_dx0;
        Matrix2f sol0 = -tmp * sol1;

        G = abs(det(-sol0));
    } else {
        // Invert single 2x2 matrix
        Float determinant = det(dc1_dx1);
        if (abs(determinant) < 1e-6f) {
            return 0.f;
        }
        Matrix2f inv_dc1_dx1 = inverse(dc1_dx1);
        Float dx1_dx2 = abs(det(inv_dc1_dx1 * dc1_dx2));

        Vector3f d = v0.p - v1.p;
        Float inv_r2 = rcp(squared_norm(d));
        d *= sqrt(inv_r2);
        Float dw0_dx1 = abs_dot(d, v1.gn) * inv_r2;
        G = dw0_dx1 * dx1_dx2;
    }

    /* Especially in single-precision, the computation of this geometric term
       is numerically rather unstable. (Similar to the caustic case.)
       So to avoid nasty outliers we need to apply some clamping here.
       In our experiments this was only necessary for a handful of pixels in a
       full rendering.
       Note that the values of 'G' are very small here. This is expected as it
       should roughly cancel out with the sensor importance later. */
    G = min(1e-5f, G);
    return G;
}

MTS_VARIANT Float
SpecularManifoldGlints<Float, Spectrum>::lean_pdf(const SurfaceInteraction3f &si,
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

MTS_VARIANT void SpecularManifoldGlints<Float, Spectrum>::print_statistics() {
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

    Float stats_booth_avg_iterations = Float(stats_bernoulli_trial_iterations) / stats_bernoulli_trial_calls;
    std::cout << std::setw(25) << std::left << "avg. Booth iterations: "
              << std::setw(10) << std::right << stats_booth_avg_iterations << std::endl;
    std::cout << std::setw(25) << std::left << "max. Booth iterations: "
              << std::setw(10) << std::right << stats_bernoulli_trial_iterations_max << std::endl;
    std::cout << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << std::endl;
}

MTS_INSTANTIATE_CLASS(SpecularManifoldGlints)
NAMESPACE_END(mitsuba)
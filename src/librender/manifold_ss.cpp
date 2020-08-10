#include <mitsuba/render/manifold_ss.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sampler.h>
#include <iomanip>

NAMESPACE_BEGIN(mitsuba)

template<typename T>
inline void update_max(std::atomic<T> & atom, const T val) {
  for(T atom_val=atom; atom_val < val && !atom.compare_exchange_weak(atom_val, val, std::memory_order_relaxed););
}

MTS_VARIANT std::atomic<int> SpecularManifoldSingleScatter<Float, Spectrum>::stats_solver_failed(0);
MTS_VARIANT std::atomic<int> SpecularManifoldSingleScatter<Float, Spectrum>::stats_solver_succeeded(0);
MTS_VARIANT std::atomic<int> SpecularManifoldSingleScatter<Float, Spectrum>::stats_bernoulli_trial_calls(0);
MTS_VARIANT std::atomic<int> SpecularManifoldSingleScatter<Float, Spectrum>::stats_bernoulli_trial_iterations(0);
MTS_VARIANT std::atomic<int> SpecularManifoldSingleScatter<Float, Spectrum>::stats_bernoulli_trial_iterations_max(0);

MTS_VARIANT SpecularManifoldSingleScatter<Float, Spectrum>::SpecularManifoldSingleScatter(
    const Scene *scene, const SMSConfig &config) {
    m_scene = scene;
    m_config = config;
}

MTS_VARIANT SpecularManifoldSingleScatter<Float, Spectrum>::~SpecularManifoldSingleScatter() {}

MTS_VARIANT Spectrum
SpecularManifoldSingleScatter<Float, Spectrum>::specular_manifold_sampling(const SurfaceInteraction3f &si,
                                                                           ref<Sampler> sampler) const {
    ScopedPhase scope_phase(ProfilerPhase::SMSCaustics);

    /* Regarding picking the specular shapes to use:
       We chose to do the most basic thing here and just loop over all shapes
       and do one estimate each. This way we can avoid additional Monte Carlo
       noise from suboptimal shape-picking. We also experimented with other
       options, such as sampling them proportional to their visible solid angle
       from the shading point (based on a bounding sphere approximation). This
       gave slightly better or worse results depending on the specific scene
       setup.
       Picking an optimal shape (as well as the initial point on it) is an
       interesting open question for now. */

    if (unlikely(!si.is_valid())) {
        return 0.f;
    }

    // Sample emitter interaction
    EmitterInteraction ei = SpecularManifold::sample_emitter_interaction(si, m_scene->caustic_emitters_single_scatter(), sampler);

    // For each specular shape, perform one estimate
    auto shapes = m_scene->caustic_casters_single_scatter();
    if (unlikely(shapes.size() == 0)) {
        return 0.f;
    }

    Spectrum result(0.f);
    for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
        const ShapePtr specular_shape = shapes[shape_idx];

        // Sample offset normal at specular surface
        Vector3f n_offset(0.f, 0.f, 1.f);
        Float p_offset = 1.f;
        Float alpha = specular_shape->bsdf()->roughness();
        if (alpha > 0.f) {
            // Assume isotropic Beckmann distribution for the surface roughness.
            // Note that the "visible normals sampling" should not be used!
            MicrofacetDistribution<Float, Spectrum> distr(MicrofacetType::Beckmann, alpha, alpha, false);
            std::tie(n_offset, p_offset) = distr.sample(Vector3f(0.f, 0.f, 1.f), sampler->next_2d());
        }

        Spectrum value(0.f);
        if (!m_config.biased) {
            // Unbiased SMS

            // Sample a single path
            auto [success, si_final, unused] = sample_path(specular_shape, si, ei, sampler, n_offset);
            if (!success) {
                stats_solver_failed++;
                continue;
            }
            stats_solver_succeeded++;
            Vector3f direction = normalize(si_final.p - si.p);

            // We already now there is no obstacle between si & si_final, but we
            // need to explicitly check visibility between the si_final and the
            // light source.
            Ray3f ray_vis;
            if (ei.is_directional()) {
                ray_vis = Ray3f(si_final.p, ei.d, si.time, si.wavelengths);
            } else {
                Vector3f d = si_final.p - ei.p;
                Float dist = norm(d);
                d *= rcp(dist);
                ray_vis = Ray3f(ei.p, d,
                                math::RayEpsilon<Float> * (1.f + hmax(abs(ei.p))),
                                dist * (1.f - math::RayEpsilon<Float>),
                                si.time, si.wavelengths);
            }
            if (m_scene->ray_test(ray_vis)) {
                continue;
            }

            // We sampled a valid path, now compute its contribution
            Spectrum specular_val = evaluate_path_contribution(si, ei, si_final);

            // Account for BSDF at shading point
            BSDFContext ctx;
            Spectrum bsdf_val = si.bsdf()->eval(ctx, si, si.to_local(direction));

            // Now estimate the (inverse) probability of this whole process with Bernoulli trials
            Float inv_prob_estimate = 1.f;
            int iterations = 1;
            stats_bernoulli_trial_calls++;
            while (true) {
                ScopedPhase scope_phase(ProfilerPhase::SMSCausticsBernoulliTrials);

                auto [success_trial, si_final_trial, unused_trial] = sample_path(specular_shape, si, ei, sampler, n_offset);
                Vector3f direction_trial = normalize(si_final_trial.p - si.p);
                if (success_trial && abs(dot(direction, direction_trial) - 1.f) < m_config.uniqueness_threshold) {
                    break;
                }

                inv_prob_estimate += 1.f;
                iterations++;

                if (m_config.max_trials > 0 && iterations > m_config.max_trials) {
                    /* There is a tiny chance always to sample super weird paths
                       that will never occur again due to small numerical imprecisions.
                       So setting a (super conservative) threshold here can help
                       to avoid infinite loops. */
                    inv_prob_estimate = 0.f;
                    break;
                }
            }
            stats_bernoulli_trial_iterations += iterations;
            update_max(stats_bernoulli_trial_iterations_max, iterations);

            // Contribution
            value = bsdf_val * specular_val * inv_prob_estimate;

        } else {
            // Biased SMS

            std::vector<Vector3f> solutions;
            for (int m = 0; m < m_config.max_trials; ++m) {
                auto [success, si_final, unused] = sample_path(specular_shape, si, ei, sampler, n_offset);
                if (!success) {
                    stats_solver_failed++;
                    continue;
                }
                stats_solver_succeeded++;
                Vector3f direction = normalize(si_final.p - si.p);

                // We already now there is no obstacle between si & si_final, but we
                // need to explicitly check visibility between the si_final and the
                // light source.
                Ray3f ray_vis;
                if (ei.is_directional()) {
                    ray_vis = Ray3f(si_final.p, ei.d, si.time, si.wavelengths);
                } else {
                    Vector3f d = si_final.p - ei.p;
                    Float dist = norm(d);
                    d *= rcp(dist);
                    ray_vis = Ray3f(ei.p, d,
                                    math::RayEpsilon<Float> * (1.f + hmax(abs(ei.p))),
                                    dist * (1.f - math::RayEpsilon<Float>),
                                    si.time, si.wavelengths);
                }
                if (m_scene->ray_test(ray_vis)) {
                    continue;
                }

                // Check if this is a new and unique solution
                bool duplicate = false;
                for (size_t k = 0; k < solutions.size(); ++k) {
                    if (abs(dot(direction, solutions[k]) - 1.f) < m_config.uniqueness_threshold) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) {
                    continue;
                }

                // Record this new solution
                solutions.push_back(direction);

                // Account for BSDF at shading point
                BSDFContext ctx;
                Spectrum bsdf_val = si.bsdf()->eval(ctx, si, si.to_local(direction));

                // Contribution
                Spectrum specular_val = evaluate_path_contribution(si, ei, si_final);
                value += bsdf_val * specular_val;
            }
        }
        result += value / p_offset;
    }
    return result;
}

MTS_VARIANT std::tuple<typename SpecularManifoldSingleScatter<Float, Spectrum>::Mask,
                       typename SpecularManifoldSingleScatter<Float, Spectrum>::SurfaceInteraction3f,
                       typename SpecularManifoldSingleScatter<Float, Spectrum>::SurfaceInteraction3f>
SpecularManifoldSingleScatter<Float, Spectrum>::sample_path(const ShapePtr shape,
                                                            const SurfaceInteraction3f &si,
                                                            const EmitterInteraction &ei,
                                                            ref<Sampler> sampler,
                                                            const Vector3f &n_offset,
                                                            const Point3f &p_start) const {
    /* Regarding sampling initial points on the shape:
       The paper describes this process very generalized as just sampling some
       initial position based on a 2d random sample but are various options
       of doing this of course.
       Implemented here is the most simple form that uniformly samples on the
       surface of the shape.
       During development we also experimented with sampling directly inside
       the cone of directions defined by the bounding sphere, which performed
       slightly better or worse depending on the scene specific setup. */

    // Sample position uniformly on shape
    auto ps = shape->sample_position(si.time, sampler->next_2d());
    if (any(neq(p_start, 0.f))) {
        ps.p = p_start;
    }
    Vector3f d_tmp = normalize(ps.p - si.p);

    // If requested, override with MNEE initialization without randomization.
    if (m_config.mnee_init) {
        Complex<Spectrum> ior = shape->bsdf()->ior(si);
        Mask reflection = any(neq(0.f, imag(ior)));
        if (reflection) {
            // Modified "MNEE"
            BoundingBox3f bbox = shape->bbox();
            d_tmp = normalize(bbox.center() - si.p);   // Modified MNEE for reflection.
        } else {
            // Standard MNEE
            d_tmp = normalize(ei.p - si.p);
        }
    }

    // Do a first ray-trace to make sure we can actually hit the target shape
    Ray3f ray_tmp(si.p, d_tmp, si.time, si.wavelengths);
    SurfaceInteraction3f si_init = m_scene->ray_intersect(ray_tmp);
    if (!si_init.is_valid() || shape != si_init.shape) {
        SurfaceInteraction3f si_dummy;
        return std::make_tuple(false, si_dummy, si_dummy);
    }

    ManifoldVertex vtx_init(si_init, m_config.twostage ? 1.f : 0.f);

    // If requested, run the "two-stage" solver for normal-mapped geometry
    if (m_config.twostage) {
        // Sample an offset normal from a gaussian defined by the LEAN information, around the current offset normal
        Point2f mu_offset(-n_offset[0]/n_offset[2], -n_offset[1]/n_offset[2]);
        auto [mu, sigma] = si_init.bsdf()->lean(si_init, true);
        Point2f slope = SpecularManifold::sample_gaussian(0.5f*(mu+mu_offset), sigma, sampler->next_2d());
        Normal3f lean_normal_local = normalize(Normal3f(-slope[0], -slope[1], 1.f));

        // First run the solver on the smoothed version of the shape without normal map.
        // This will bring us close to the solutions on the actual shape.
        auto [success_smooth, si_smooth] = newton_solver(si, vtx_init, ei, lean_normal_local, 1.f);
        if (success_smooth && si_smooth.is_valid()) {
            vtx_init = ManifoldVertex(si_smooth, 0.f);
        } else {
            SurfaceInteraction3f si_dummy;
            return std::make_tuple(false, si_dummy, si_dummy);
        }
    }

    // Run the newton solver to find a valid solution vertex
    auto [success, si_final] = newton_solver(si, vtx_init, ei, n_offset, 0.f);
    if (!success || !si_final.is_valid()) {
        return std::make_tuple(false, si_final, si_init);
    }

    return std::make_tuple(true, si_final, si_init);
}

MTS_VARIANT Spectrum
SpecularManifoldSingleScatter<Float, Spectrum>::evaluate_path_contribution(const SurfaceInteraction3f &si,
                                                                           const EmitterInteraction &ei_,
                                                                           const SurfaceInteraction3f &si_final) const {
    // Specular point to ManifoldVertex
    ManifoldVertex vtx(si_final);
    vtx.make_orthonormal();

    // Emitter to ManifoldVertex
    EmitterInteraction ei(ei_);
    if (ei.is_point()) {
        /* Due to limitations in the API (e.g. no way to evaluate discrete emitter
           distributions directly), we need to re-sample this class of emitters
           one more time, in case they have some non-uniform emission profile
           (e.g. spotlights). It all works out because we're guaranteed to sample
           the same (delta) position again though and we don't even need a random
           sample. */
        auto [ds, spec] = ei.emitter->sample_direction(si_final, Point2f(0.f));
        ei.p = ds.p;
        ei.d = ds.d;
        ei.n = ei.d;
        /* Because this is a delta light, ei.pdf just stores the discrete prob.
           of picking the initial emitter. It's important that we don't lose
           this quantity here when re-calculating the intensity. */
        Float emitter_pdf = ei.pdf;
        ei.pdf = ds.pdf;
        // Remove solid angle conversion factor. This is accounted for in the geometric term.
        ei.weight = spec * ds.dist * ds.dist;

        ei.pdf *= emitter_pdf;
        ei.weight *= rcp(emitter_pdf);
    }
    auto[sucess_e, vy] = SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, vtx.p, si.time, si.wavelengths);
    if (!sucess_e) {
        return 0.f;
    }

    // Shading point to ManifoldVertex
    ManifoldVertex vx = ManifoldVertex(si);
    vx.make_orthonormal();

    Spectrum path_throughput(1.f);
    path_throughput *= specular_reflectance(si_final, normalize(ei.p - si_final.p));
    path_throughput *= geometric_term(vx, vtx, vy);
    path_throughput *= ei.weight;
    return path_throughput;
}

MTS_VARIANT std::pair<typename SpecularManifoldSingleScatter<Float, Spectrum>::Mask,
                      typename SpecularManifoldSingleScatter<Float, Spectrum>::SurfaceInteraction3f>
SpecularManifoldSingleScatter<Float, Spectrum>::newton_solver(const SurfaceInteraction3f &si,
                                                              const ManifoldVertex &vtx_init,
                                                              const EmitterInteraction &ei,
                                                              const Vector3f &n_offset,
                                                              Float smoothing) const {
    ManifoldVertex vtx = vtx_init;

    // Newton iterations..
    bool success = false;
    size_t iterations = 0;
    Float beta = 1.f;

    SurfaceInteraction3f si_current;
    while (iterations < m_config.max_iterations) {
        bool step_success;
        Vector2f C;
        Vector2f dX;
        if (m_config.halfvector_constraints) {
            // Use standard manifold formulation using half-vector constraints
            std::tie(step_success, C, dX) = compute_step_halfvector(si.p, vtx, ei, n_offset);
        } else {
            // Use angle-difference constraint formulation
            std::tie(step_success, C, dX) = compute_step_anglediff(si.p, vtx, ei, n_offset);
        }
        if (!step_success) {
            break;
        }

        // Check for success
        if (norm(C) < m_config.solver_threshold) {
            success = true;
            break;
        }

        // Make a proposal
        Vector3f p_prop = vtx.p - m_config.step_scale*beta * (vtx.dp_du * dX[0] + vtx.dp_dv * dX[1]);

        // Project back to surfaces
        Vector3f d_prop = normalize(p_prop - si.p);
        Ray3f ray_prop(si.p, d_prop, si.time, si.wavelengths);
        si_current = m_scene->ray_intersect(ray_prop);
        if (!si_current.is_valid() ||
            vtx.shape != si_current.shape) {
            // Missed scene completely or hit different shape, need smaller step!
            beta = 0.5f*beta;
            iterations += 1;
            continue;
        }

        beta = std::min(Float(1), Float(2)*beta);
        vtx = ManifoldVertex(si_current, smoothing);

        iterations++;
    }

    if (!success) {
        return { false, si_current };
    }

    /* In the refraction case, the half-vector formulation of Manifold
       walks will often converge to invalid solutions that are actually
       reflections. Here we need to reject those. */
    Vector3f wx = normalize(si.p - vtx.p);
    Vector3f wy = ei.is_directional() ? ei.d : normalize(ei.p - vtx.p);
    Float cos_theta_x = dot(vtx.gn, wx),
          cos_theta_y = dot(vtx.gn, wy);
    bool refraction = cos_theta_x * cos_theta_y < 0.f;
    bool reflection = !refraction;
    if ((vtx.eta == 1.f && !reflection) ||
        (vtx.eta != 1.f && !refraction)) {
        return { false, si_current };
    }

    return { true, si_current };
}

MTS_VARIANT std::tuple<typename SpecularManifoldSingleScatter<Float, Spectrum>::Mask,
                       typename SpecularManifoldSingleScatter<Float, Spectrum>::Vector2f,
                       typename SpecularManifoldSingleScatter<Float, Spectrum>::Vector2f>
SpecularManifoldSingleScatter<Float, Spectrum>::compute_step_halfvector(const Point3f &v0p,
                                                                        const ManifoldVertex &v1,
                                                                        const EmitterInteraction &v2,
                                                                        const Vector3f &n_offset) const {
    // Setup wi / wo
    Vector3f wo;
    if (v2.is_directional()) {
        // Case of fixed 'wo' direction
        wo = v2.d;
    } else {
        // Standard case for fixed emitter position
        wo = v2.p - v1.p;
    }
    Float ilo = norm(wo);
    if (ilo < 1e-3f) {
        return std::make_tuple(false, Vector2f(math::Infinity<Float>), Vector2f(0.f));
    }
    ilo = rcp(ilo);
    wo *= ilo;

    Vector3f wi = v0p - v1.p;
    Float ili = norm(wi);
    if (ili < 1e-3f) {
        return std::make_tuple(false, Vector2f(math::Infinity<Float>), Vector2f(0.f));
    }
    ili = rcp(ili);
    wi *= ili;

    // Setup generalized half-vector
    Float eta = v1.eta;
    if (dot(wi, v1.gn) < 0.f) {
        eta = rcp(eta);
    }
    Vector3f h = wi + eta * wo;
    if (eta != 1.f) h *= -1.f;
    Float ilh = rcp(norm(h));
    h *= ilh;

    ilo *= eta * ilh;
    ili *= ilh;

    // Derivative of specular constraint w.r.t. v1
    Vector3f dh_du, dh_dv;
    if (v2.is_directional()) {
        // When the 'wo' direction is fixed, the derivative here simplifies.
        dh_du = ili * (-v1.dp_du + wi * dot(wi, v1.dp_du));
        dh_dv = ili * (-v1.dp_dv + wi * dot(wi, v1.dp_dv));
    } else {
        // Standard case for fixed emitter position
        dh_du = -v1.dp_du * (ili + ilo) + wi * (dot(wi, v1.dp_du) * ili)
                                        + wo * (dot(wo, v1.dp_du) * ilo);
        dh_dv = -v1.dp_dv * (ili + ilo) + wi * (dot(wi, v1.dp_dv) * ili)
                                        + wo * (dot(wo, v1.dp_dv) * ilo);
    }
    dh_du -= h * dot(dh_du, h);
    dh_dv -= h * dot(dh_dv, h);
    if (eta != 1.f) {
        dh_du *= -1.f;
        dh_dv *= -1.f;
    }

    Matrix2f dH_dX(0.f);
    dH_dX(0,0) = dot(v1.ds_du, h) + dot(v1.s, dh_du);
    dH_dX(1,0) = dot(v1.dt_du, h) + dot(v1.t, dh_du);
    dH_dX(0,1) = dot(v1.ds_dv, h) + dot(v1.s, dh_dv);
    dH_dX(1,1) = dot(v1.dt_dv, h) + dot(v1.t, dh_dv);

    // Invert matrix
    Float determinant = det(dH_dX);
    if (abs(determinant) < 1e-6f) {
        return std::make_tuple(false, Vector2f(math::Infinity<Float>), Vector2f(0.f));
    }
    Matrix2f dX_dH = inverse(dH_dX);

    // Evaluate constraint
    Vector2f H(dot(v1.s, h), dot(v1.t, h));
    Vector2f N(n_offset[0], n_offset[1]);
    Vector2f dH = H - N;

    // Compute step
    Vector2f dX = dX_dH * dH;
    return std::make_tuple(true, dH, dX);
}

MTS_VARIANT std::tuple<typename SpecularManifoldSingleScatter<Float, Spectrum>::Mask,
                       typename SpecularManifoldSingleScatter<Float, Spectrum>::Vector2f,
                       typename SpecularManifoldSingleScatter<Float, Spectrum>::Vector2f>
SpecularManifoldSingleScatter<Float, Spectrum>::compute_step_anglediff(const Point3f &v0p,
                                                                       const ManifoldVertex &v1,
                                                                       const EmitterInteraction &v2,
                                                                       const Vector3f &n_offset) const {
    // wi / wo & their derivatives
    Vector3f wi = v0p - v1.p;
    Float ili = norm(wi);
    if (ili < 1e-3f) {
        return std::make_tuple(false, Vector2f(math::Infinity<Float>), Vector2f(0.f));
    }
    ili = rcp(ili);
    wi *= ili;

    Vector3f dwi_du = -ili * (v1.dp_du - wi*dot(wi, v1.dp_du)),
             dwi_dv = -ili * (v1.dp_dv - wi*dot(wi, v1.dp_dv));

    Vector3f wo;
    if (v2.is_directional()) {
        // Case of fixed 'wo' direction
        wo = v2.d;
    } else {
        // Standard case for fixed emitter position
        wo = v2.p - v1.p;
    }
    Float ilo = norm(wo);
    if (ilo < 1e-3f) {
        return std::make_tuple(false, Vector2f(math::Infinity<Float>), Vector2f(0.f));
    }
    ilo = rcp(ilo);
    wo *= ilo;

    Vector3f dwo_du, dwo_dv;
    if (v2.is_directional()) {
        // Fixed 'wo' direction means its derivatives must be zero
        dwo_du = Vector3f(0.f);
        dwo_dv = Vector3f(0.f);
    } else {
        // Standard case for fixed emitter position
        dwo_du = -ilo * (v1.dp_du - wo*dot(wo, v1.dp_du));
        dwo_dv = -ilo * (v1.dp_dv - wo*dot(wo, v1.dp_dv));
    }

    // Set up constraint function and its derivatives
    Vector2f C(0.f);
    Matrix2f dC_dX(0.f);

    auto transform = [&](const Vector3f &w, const Vector3f &n, Float eta) {
        if (eta == 1.f) {
            return SpecularManifold::reflect(w, n);
        } else {
            return SpecularManifold::refract(w, n, eta);
        }
    };
    auto d_transform = [&](const Vector3f &w, const Vector3f &dw_du, const Vector3f &dw_dv,
                           const Vector3f &n, const Vector3f &dn_du, const Vector3f &dn_dv,
                           Float eta) {
        if (eta == 1.f) {
            return SpecularManifold::d_reflect(w, dw_du, dw_dv, n, dn_du, dn_dv);
        } else {
            return SpecularManifold::d_refract(w, dw_du, dw_dv, n, dn_du, dn_dv, eta);
        }
    };

    // Handle offset normal. These are no-ops in case n_offset=[0,0,1]
    Normal3f n = v1.s * n_offset[0] +
                 v1.t * n_offset[1] +
                 v1.n * n_offset[2];
    Vector3f dn_du = v1.ds_du * n_offset[0] +
                     v1.dt_du * n_offset[1] +
                     v1.dn_du * n_offset[2];
    Vector3f dn_dv = v1.ds_dv * n_offset[0] +
                     v1.dt_dv * n_offset[1] +
                     v1.dn_dv * n_offset[2];

    bool success = false;

    auto [valid_i_refr_i, wio] = transform(wi, n, v1.eta);
    if (valid_i_refr_i) {
        auto [dwio_du, dwio_dv] = d_transform(wi, dwi_du, dwi_dv, n, dn_du, dn_dv, v1.eta);
        auto [to, po]   = SpecularManifold::sphcoords(wo);
        auto [tio, pio] = SpecularManifold::sphcoords(wio);
        C = Vector2f(to - tio, po - pio);

        auto [dto_du, dpo_du, dto_dv, dpo_dv]     = SpecularManifold::d_sphcoords(wo, dwo_du, dwo_dv);
        auto [dtio_du, dpio_du, dtio_dv, dpio_dv] = SpecularManifold::d_sphcoords(wio, dwio_du, dwio_dv);

        dC_dX(0,0) = dto_du - dtio_du;
        dC_dX(1,0) = dpo_du - dpio_du;
        dC_dX(0,1) = dto_dv - dtio_dv;
        dC_dX(1,1) = dpo_dv - dpio_dv;

        success = true;
    }

    auto [valid_o_refr_o, woi] = transform(wo, n, v1.eta);
    if (valid_o_refr_o && !success) {
        auto [dwoi_du, dwoi_dv] = d_transform(wo, dwo_du, dwo_dv, n, dn_du, dn_dv, v1.eta);

        auto [ti, pi]   = SpecularManifold::sphcoords(wi);
        auto [toi, poi] = SpecularManifold::sphcoords(woi);
        C = Vector2f(ti - toi, pi - poi);

        auto [dti_du, dpi_du, dti_dv, dpi_dv]     = SpecularManifold::d_sphcoords(wi, dwi_du, dwi_dv);
        auto [dtoi_du, dpoi_du, dtoi_dv, dpoi_dv] = SpecularManifold::d_sphcoords(woi, dwoi_du, dwoi_dv);

        dC_dX(0,0) = dti_du - dtoi_du;
        dC_dX(1,0) = dpi_du - dpoi_du;
        dC_dX(0,1) = dti_dv - dtoi_dv;
        dC_dX(1,1) = dpi_dv - dpoi_dv;

        success = true;
    }

    // Invert matrix
    Float determinant = det(dC_dX);
    if (abs(determinant) < 1e-6f) {
        return std::make_tuple(false, Vector2f(math::Infinity<Float>), Vector2f(0.f));
    }
    Matrix2f dX_dC = inverse(dC_dX);

    // Compute step
    Vector2f dX = dX_dC * C;
    return std::make_tuple(success, C, dX);
}

MTS_VARIANT Spectrum
SpecularManifoldSingleScatter<Float, Spectrum>::specular_reflectance(const SurfaceInteraction3f &si,
                                                                     const Vector3f &wo) const {
    if (!si.is_valid()) return 0.f;
    const BSDFPtr bsdf = si.shape->bsdf();
    if (!bsdf) return 0.f;

    Complex<Spectrum> ior = si.bsdf()->ior(si);
    Mask reflection = any(neq(0.f, imag(ior)));

    Spectrum bsdf_val(1.f);
    if (bsdf->roughness() > 0.f) {
        // Glossy BSDF: evaluate BSDF and transform to half-vector domain.
        BSDFContext ctx;
        Vector3f wo_l = si.to_local(wo);
        bsdf_val = bsdf->eval(ctx, si, wo_l);

        /* Compared to Eq. 6 in [Hanika et al. 2015 (MNEE)], two terms are omitted:
           1) abs_dot(wo, n) is part of BSDF::eval
           2) abs_dot(h, n)  is part of the Microfacet distr. (also in BSDF::eval) */
        Vector3f h_l;
        if (reflection) {
            h_l = normalize(si.wi + wo_l);
            bsdf_val *= 4.f*abs_dot(wo_l, h_l);
        } else {
            Float eta = hmean(real(ior));
            h_l = -normalize(si.wi + eta*wo_l);
            bsdf_val *= sqr(dot(si.wi, h_l) + eta*dot(wo_l, h_l)) / (eta*eta * abs_dot(wo_l, h_l));
        }
    } else {
        // Delta BSDF: just account for Fresnel term and solid angle compression
        Frame3f frame = bsdf->frame(si, 0.f);
        Float cos_theta = dot(frame.n, wo);
        if (reflection) {
            if (all(eq(imag(ior), 0.f))) {
                auto [F_, cos_theta_t, eta_it, eta_ti] = fresnel(Spectrum(abs(cos_theta)), real(ior));
                bsdf_val = F_;
            } else {
                bsdf_val = fresnel_conductor(Spectrum(abs(cos_theta)), ior);
            }
        } else {
            Float eta = hmean(real(ior));
            if (cos_theta < 0.f) {
                eta = rcp(eta);
            }
            auto [F, unused_0, unused_1, unused_2] = fresnel(cos_theta, eta);
            bsdf_val = 1.f - F;
            bsdf_val *= sqr(eta);
        }
    }

    return bsdf_val;
}

MTS_VARIANT Float
SpecularManifoldSingleScatter<Float, Spectrum>::geometric_term(const ManifoldVertex &v0,
                                                               const ManifoldVertex &v1,
                                                               const ManifoldVertex &v2) const {
    // A lot of the computation here overlaps with 'compute_step_halfvector', but
    // we need to do a bit more work involving the endpoints to get the full
    // geometric term.

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
    Float eta = v1.eta;
    if (dot(wi, v1.gn) < 0.f) {
        eta = rcp(eta);
    }
    Vector3f h = wi + eta * wo;
    if (eta != 1.f) h *= -1.f;
    Float ilh = rcp(norm(h));
    h *= ilh;

    ilo *= eta * ilh;
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
        if (eta != 1.f) {
            dh_du *= -1.f;
            dh_dv *= -1.f;
        }
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
    if (eta != 1.f) {
        dh_du *= -1.f;
        dh_dv *= -1.f;
    }

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
    if (eta != 1.f) {
        dh_du *= -1.f;
        dh_dv *= -1.f;
    }
    Matrix2f dc1_dx2(
        dot(dh_du, s), dot(dh_dv, s),
        dot(dh_du, t), dot(dh_dv, t)
    );

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

        Float G = abs(det(-sol0));
        /* Unfortunately, these geometric terms are very unstable, so to avoid
           severe variance we need to clamp here. */
        G = min(G, Float(10.f));
        G /= abs_dot(wi, v0.n); // Cancel out cosine term that will be added during BSDF evaluation
        G *= sqr(eta);
        return G;
    } else {
        // Invert single 2x2 matrix
        Float determinant = det(dc1_dx1);
        if (abs(determinant) < 1e-6f) {
            return 0.f;
        }
        Matrix2f inv_dc1_dx1 = inverse(dc1_dx1);
        Float dx1_dx2 = abs(det(inv_dc1_dx1 * dc1_dx2));
        /* Unfortunately, these geometric terms are very unstable, so to avoid
           severe variance we need to clamp here. */
        dx1_dx2 = min(dx1_dx2, Float(1.f));
        Vector3f d = v0.p - v1.p;
        Float inv_r2 = rcp(squared_norm(d));
        d *= sqrt(inv_r2);
        Float dw0_dx1 = abs_dot(d, v1.gn) * inv_r2;
        Float G = dw0_dx1 * dx1_dx2;
        return G;
    }
}

MTS_VARIANT void SpecularManifoldSingleScatter<Float, Spectrum>::print_statistics() {
    Float solver_success_ratio = Float(stats_solver_succeeded) / (stats_solver_succeeded + stats_solver_failed),
          solver_fail_ratio    = Float(stats_solver_failed)    / (stats_solver_succeeded + stats_solver_failed);

    std::cout << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "    Specular Manifold Sampling Statistics" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << std::setw(25) << std::left << "Solver succeeded: "
              << std::setw(10) << std::right << stats_solver_succeeded << " "
              << std::setw(8) << "(" << 100*solver_success_ratio << "%)" << std::endl;
    std::cout << std::setw(25) << std::left << "Solver failed: "
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

MTS_INSTANTIATE_CLASS(SpecularManifoldSingleScatter)
NAMESPACE_END(mitsuba)

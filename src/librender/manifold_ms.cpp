#include <mitsuba/render/manifold_ms.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sampler.h>
#include <iomanip>

NAMESPACE_BEGIN(mitsuba)

template<typename T>
inline void update_max(std::atomic<T> & atom, const T val) {
  for(T atom_val=atom; atom_val < val && !atom.compare_exchange_weak(atom_val, val, std::memory_order_relaxed););
}

MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_solver_failed(0);
MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_solver_succeeded(0);
MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_bernoulli_trial_calls(0);
MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_bernoulli_trial_iterations(0);
MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_bernoulli_trial_iterations_max(0);

MTS_VARIANT void
SpecularManifoldMultiScatter<Float, Spectrum>::init(const Scene *scene,
                                                    const SMSConfig &config) {
    m_scene = scene;
    m_config = config;
}

MTS_VARIANT Spectrum
SpecularManifoldMultiScatter<Float, Spectrum>::specular_manifold_sampling(const SurfaceInteraction3f &si,
                                                                          ref<Sampler> sampler) {
    ScopedPhase scope_phase(ProfilerPhase::SMSCaustics);

    /* Regarding the N-bounce implementation and the seed path sampling:
       Here it is even more tricky to choose a good initial guess for the Newton
       solver to use.
       You could think of various strategies of sampling N specular points
       (e.g. uniformly distributed over specular surfaces like in the
       single-bounce case).
       This of course also raises the additional question of a) how to choose N
       in the first place, and b) what scattering type (reflection vs. transmission)
       should be performed at each vertex.
       In principle, SMS is compatible with all kinds of strategies here but its
       performance will greatly depend on it. E.g. the unbiased version will
       produce severe variance when we don't sample well enough and the biased
       version will suffer from energy loss instead. Coming up with smart ways
       of sampling seed paths is therefore an interesting question for potential
       future work.

       Also consider that for the PDF estimation (either unbiased or biased) to
       work, we need a discrete, finite set of solutions to enumerate
       stochastically. This might limit the range of possible strategies.

       For the glossy case, we further need to sample one set of offset normals
       for each rough interface that we sample. And these need to stay constant
       during the PDF estimation process (otherwise we have a continuum of
       infinitely many solutions and the estimate doesn't work anymore).
       This also means the shapes involved in the path need to stay the same.

       In this specific implementation we opted for the following (simple)
       strategy:
       1) We fix the number of bounces.
       2) We first sample a uniformly random point on shapes marked as "multi
          caustic casters".
       3) We trace a ray through those point, reflecting on (marked) conductors
          and refracting on dielectrics that we hit along the way.
       3b) Optional: whenever we hit a glossy shape in the first iteration of
           doing this, also sample an offset normal for it.
       4) After we hit the number of desired specular bounces, we do a straight
          connection to the light source. There might be a visibility issue here
          but we will only check for it once the final path is found.

       Whenever we sample more paths (for PDF estimation) we now immediately
       reject the path if it lies on different shapes (otherwise the sampled
       offset normals don't make sense).
       */

    if (unlikely(!si.is_valid())) {
        return 0.f;
    }

    // Sample emitter interaction
    EmitterInteraction ei = SpecularManifold::sample_emitter_interaction(si, m_scene->caustic_emitters_multi_scatter(), sampler);

    // For each specular shape, perform one estimate
    auto shapes = m_scene->caustic_casters_multi_scatter();
    if (unlikely(shapes.size() == 0)) {
        return 0.f;
    }

    Spectrum result(0.f);
    for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
        const ShapePtr specular_shape = shapes[shape_idx];

        m_offset_normals_pdf = 1.f;

        Spectrum value(0.f);
        if (!m_config.biased) {
            // Unbiased SMS

            // Sample a single path
            bool success = sample_path(specular_shape, si, ei, sampler, true);
            if (!success) {
                stats_solver_failed++;
                continue;
            }
            stats_solver_succeeded++;
            Vector3f direction = normalize(m_current_path[0].p - si.p);

            // We sampled a valid path, now compute its contribution. This also checks for visibility.
            Spectrum specular_val = evaluate_path_contribution(si, ei);

            // Account for BSDF at shading point
            BSDFContext ctx;
            Spectrum bsdf_val = si.bsdf()->eval(ctx, si, si.to_local(direction));

            // Now estimate the (inverse) probability of this whole process with Bernoulli trials
            Float inv_prob_estimate = 1.f;
            int iterations = 1;
            stats_bernoulli_trial_calls++;
            while (true) {
                ScopedPhase scope_phase(ProfilerPhase::SMSCausticsBernoulliTrials);

                bool success_trial = sample_path(specular_shape, si, ei, sampler, false);
                Vector3f direction_trial = normalize(m_current_path[0].p - si.p);
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

            /* The biased version with roughness is a bit weird with when the
               offset normals get sampled. They need to be consistent during the
               PDF estimation process so ideally we would just sample them ahead
               of time here.
               However, due to the way we generate the seed path, we don't even
               know which surfaces will all be part of the path so sampling based
               on their roughness is not yet possible.
               As a workaround, the first complete seed path will sample them
               and later serve as a "template" which means following paths need
               to interact with the same shapes. With possible improved sampling
               strategies in the future, this detail might need to be reconsidered. */
            m_seed_path.clear();

            std::vector<Vector3f> solutions;
            for (int m = 0; m < m_config.max_trials; ++m) {
                // Sample path, but only sample offset normals when we couldn't establish a seed path yet
                bool success = sample_path(specular_shape, si, ei, sampler, m_seed_path.size() == 0);
                if (!success) {
                    stats_solver_failed++;
                    continue;
                }

                stats_solver_succeeded++;
                Vector3f direction = normalize(m_current_path[0].p - si.p);

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
                Spectrum specular_val = evaluate_path_contribution(si, ei);
                value += bsdf_val * specular_val;
            }
        }

        result += value / m_offset_normals_pdf;
    }
    return result;
}

MTS_VARIANT typename SpecularManifoldMultiScatter<Float, Spectrum>::Mask
SpecularManifoldMultiScatter<Float, Spectrum>::sample_path(const ShapePtr shape,
                                                           const SurfaceInteraction3f &si,
                                                           const EmitterInteraction &ei,
                                                           ref<Sampler> sampler,
                                                           bool first_path,
                                                           const Point3f &p_start) {
    /* To keep complexity reasonable, the two-stage variation is omitted
       for this multi-bounce implementation but could be added with a few
       generalizations. */

    Mask success_seed = sample_seed_path(shape, si, ei, sampler, first_path, p_start);
    if (!success_seed) {
        return false;
    }

    // return false;

    Mask success_solve = newton_solver(si, ei);
    if (!success_solve) {
        return false;
    }

    return true;
}

MTS_VARIANT Spectrum
SpecularManifoldMultiScatter<Float, Spectrum>::evaluate_path_contribution(const SurfaceInteraction3f &si,
                                                                          const EmitterInteraction &ei_) {
    ManifoldVertex &vtx_last = m_current_path[m_current_path.size() - 1];

    // Emitter to ManifoldVertex
    EmitterInteraction ei(ei_);
    if (ei.is_point()) {
        /* Due to limitations in the API (e.g. no way to evaluate discrete emitter
           distributions directly), we need to re-sample this class of emitters
           one more time, in case they have some non-uniform emission profile
           (e.g. spotlights). It all works out because we're guaranteed to sample
           the same (delta) position again though and we don't even need a random
           sample. */
        SurfaceInteraction3f si_last(si);
        si_last.p = vtx_last.p; si_last.n = vtx_last.gn; si_last.sh_frame.n = vtx_last.n;
        auto [ds, spec] = ei.emitter->sample_direction(si_last, Point2f(0.f));
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
    auto[sucess_e, vy] = SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, vtx_last.p, si.time, si.wavelengths);
    if (!sucess_e) {
        return 0.f;
    }

    for (size_t k = 0; k < m_current_path.size(); ++k) {
        m_current_path[k].make_orthonormal();
    }

    // Shading point to ManifoldVertex
    ManifoldVertex vx(si, 0.f);
    vx.make_orthonormal();

    Spectrum path_throughput(1.f);
    path_throughput *= specular_reflectance(si, ei);
    path_throughput *= geometric_term(vx, vy);
    path_throughput *= ei.weight;
    return path_throughput;
}

MTS_VARIANT typename SpecularManifoldMultiScatter<Float, Spectrum>::Mask
SpecularManifoldMultiScatter<Float, Spectrum>::sample_seed_path(const ShapePtr shape,
                                                                const SurfaceInteraction3f &si_,
                                                                const EmitterInteraction &ei,
                                                                ref<Sampler> sampler,
                                                                bool first_path,
                                                                const Point3f &p_start) {
    m_current_path.clear();
    if (first_path) {
        m_seed_path.clear();
        m_offset_normals.clear();
        m_offset_normals_pdf = 1.f;
    }

    SurfaceInteraction3f si(si_);

    // Sample first point on specular shape directly ...
    Point3f x0 = si.p;
    auto ps = shape->sample_position(si.time, sampler->next_2d());
    Point3f x1 = ps.p;
    if (any(neq(p_start, 0.f))) {
        x1 = p_start;
    }
    Vector3f wo = normalize(x1 - x0);

    // If requested, override with MNEE initialization without randomization.
    if (m_config.mnee_init) {
        Complex<Spectrum> ior = shape->bsdf()->ior(si);
        Mask reflection = any(neq(0.f, imag(ior)));
        if (reflection) {
            // Modified "MNEE"
            BoundingBox3f bbox = shape->bbox();
            wo = normalize(bbox.center() - x0);   // Modified MNEE for reflection.
        } else {
            // Standard MNEE
            wo = normalize(ei.p - x0);
        }
    }

    Ray3f ray(x0, wo, si.time, si.wavelengths);

    while (true) {
        int bounce = m_current_path.size();

        if (bounce >= m_config.bounces) {
            /* We reached the number of specular bounces that was requested.
               (Implicitly) connect to the light source now by terminating. */
            break;
        }

        si = m_scene->ray_intersect(ray);
        if (!si.is_valid()) {
            return false;
        }
        const ShapePtr shape = si.shape;
        if (!shape->is_caustic_caster_multi_scatter() &&
            !shape->is_caustic_bouncer()) {
            // We intersected something that cannot be par of a specular chain
            return false;
        }

        // On first intersect, make sure that we actually hit this target shape.
        if (bounce == 0 && (shape != shape)) {
            return false;
        }

        if (!first_path && shape != m_seed_path[bounce].shape) {
            /* When we sample multiple paths for the PDF estimation process,
               be conservative and only allow paths that intersect the same
               shapes again. */
            return false;
        }

        // Create the path vertex
        ManifoldVertex vertex = ManifoldVertex(si, 0.f);
        m_current_path.push_back(vertex);

        // Potentially sample an offset normal here.
        Vector3f n_offset;
        if (first_path) {
            // This is the first seed path (before potential Bernoulli trials).
            n_offset = Vector3f(0.f, 0.f, 1.f);
            Float p_o = 1.f;

            Float alpha = shape->bsdf()->roughness();
            if (alpha > 0.f) {
                // Assume isotropic Beckmann distribution for the surface roughness.
                MicrofacetDistribution<Float, Spectrum> distr(MicrofacetType::Beckmann, alpha, alpha, false);
                std::tie(n_offset, p_o) = distr.sample(Vector3f(0.f, 0.f, 1.f), sampler->next_2d());
            }
            m_offset_normals_pdf *= p_o;
            m_offset_normals.push_back(n_offset);
        } else {
            // An offset normal was already sampled previously and we need to stick to it.
            n_offset = m_offset_normals[bounce];
        }

        // Perform scattering at vertex, unless we are doing the straight-line MNEE initialization
        if (!m_config.mnee_init) {
            // Get current (potentially offset) normal in world space
            Vector3f m = vertex.s * n_offset[0] +
                         vertex.t * n_offset[1] +
                         vertex.n * n_offset[2];

            Vector3f wi = -wo;
            bool scatter_success = false;
            if (vertex.eta == 1.f) {
                std::tie(scatter_success, wo) = SpecularManifold::reflect(wi, m);
            } else {
                std::tie(scatter_success, wo) = SpecularManifold::refract(wi, m, vertex.eta);
            }

            if (!scatter_success) {
                // We must have hit total internal reflection. Abort.
                return false;
            }
        }

        ray = si.spawn_ray(wo);
    }

    if (first_path) {
        /* This is the first time we sampled a seed path. Keep it around for
           potential Bernoulli trials later. */
        m_seed_path = m_current_path;
    }

    return true;
}

MTS_VARIANT typename SpecularManifoldMultiScatter<Float, Spectrum>::Mask
SpecularManifoldMultiScatter<Float, Spectrum>::newton_solver(const SurfaceInteraction3f &si,
                                                             const EmitterInteraction &ei) {
    // Newton iterations..
    bool success = false;
    size_t iterations = 0;
    Float beta = 1.f;

    bool needs_step_update = true;
    while (iterations < m_config.max_iterations) {
        bool step_success = true;
        if (needs_step_update) {
            if (m_config.halfvector_constraints) {
                // Use standard manifold formulation using half-vector constraints
                step_success = compute_step_halfvector(si.p, ei);
            } else {
                // Use angle-difference constraint formulation
                step_success = compute_step_anglediff(si.p, ei);
            }
        }
        if (!step_success) {
            break;
        }

        // Check for success
        bool converged = true;
        for (size_t i = 0; i < m_current_path.size(); ++i) {
            const ManifoldVertex &v = m_current_path[i];
            if (norm(v.C) > m_config.solver_threshold) {
                converged = false;
                break;
            }
        }
        if (converged) {
            success = true;
            break;
        }

        // Make a proposal
        m_proposed_positions.clear();
        for (size_t i = 0; i < m_current_path.size(); ++i) {
            const ManifoldVertex &v = m_current_path[i];
            Point3f p_prop = v.p - m_config.step_scale*beta * (v.dp_du * v.dx[0] + v.dp_dv * v.dx[1]);
            m_proposed_positions.push_back(p_prop);
        }

        // Project back to surfaces
        bool project_success = reproject(si);
        if (!project_success) {
            beta = 0.5f*beta;
            needs_step_update = false;
        } else {
            beta = std::min(Float(1), Float(2)*beta);
            m_current_path = m_proposed_path;
            needs_step_update = true;
        }

        iterations++;
    }

    if (!success) {
        return false;
    }

    /* In the refraction case, the half-vector formulation of Manifold
       walks will often converge to invalid solutions that are actually
       reflections. Here we need to reject those. */
    size_t n = m_current_path.size();
    for (size_t i = 0; i < n; ++i) {
        Point3f x_prev = (i == 0)   ? si.p : m_current_path[i-1].p;
        Point3f x_next = (i == n-1) ? ei.p : m_current_path[i+1].p;
        Point3f x_cur  = m_current_path[i].p;

        bool at_endpoint_with_fixed_direction = (i == (n-1) && ei.is_directional());
        Vector3f wi = normalize(x_prev - x_cur);
        Vector3f wo = at_endpoint_with_fixed_direction ? ei.d : normalize(x_next - x_cur);

        Float cos_theta_i = dot(m_current_path[i].gn, wi),
              cos_theta_o = dot(m_current_path[i].gn, wo);
        bool refraction = cos_theta_i * cos_theta_o < 0.f,
             reflection = !refraction;
        if ((m_current_path[i].eta == 1.f && !reflection) ||
            (m_current_path[i].eta != 1.f && !refraction)) {
            return false;
        }
    }

    return true;
}

MTS_VARIANT typename SpecularManifoldMultiScatter<Float, Spectrum>::Mask
SpecularManifoldMultiScatter<Float, Spectrum>::compute_step_halfvector(const Point3f &x0,
                                                                       const EmitterInteraction &ei) {
    std::vector<ManifoldVertex> &v = m_current_path;

    size_t k = v.size();
    for (size_t i = 0; i < k; ++i) {
        v[i].C = Vector2f(0.f);
        v[i].dC_dx_prev = Matrix2f(0.f);
        v[i].dC_dx_cur  = Matrix2f(0.f);
        v[i].dC_dx_next = Matrix2f(0.f);

        Point3f x_prev = (i == 0)   ? x0   : v[i-1].p;
        Point3f x_next = (i == k-1) ? ei.p : v[i+1].p;
        Point3f x_cur  = v[i].p;

        bool at_endpoint_with_fixed_direction = (i == (k-1) && ei.is_directional());

        // Setup wi / wo
        Vector3f wo;
        if (at_endpoint_with_fixed_direction) {
            // Case of fixed 'wo' direction
            wo = ei.d;
        } else {
            // Standard case for fixed emitter position
            wo = x_next - x_cur;
        }
        Float ilo = norm(wo);
        if (ilo < 1e-3f) {
            return false;
        }
        ilo = rcp(ilo);
        wo *= ilo;

        Vector3f wi = x_prev - x_cur;
        Float ili = norm(wi);
        if (ili < 1e-3f) {
            return false;
        }
        ili = rcp(ili);
        wi *= ili;

        // Setup generalized half-vector
        Float eta = v[i].eta;
        if (dot(wi, v[i].gn) < 0.f) {
            eta = rcp(eta);
        }
        Vector3f h = wi + eta * wo;
        if (eta != 1.f) h *= -1.f;
        Float ilh = rcp(norm(h));
        h *= ilh;

        ilo *= eta * ilh;
        ili *= ilh;

        Vector3f dh_du, dh_dv;

        // Derivative of specular constraint w.r.t. x_{i-1}
        if (i > 0) {
            dh_du = ili * (v[i-1].dp_du - wi * dot(wi, v[i-1].dp_du));
            dh_dv = ili * (v[i-1].dp_dv - wi * dot(wi, v[i-1].dp_dv));

            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);
            if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            }

            v[i].dC_dx_prev = Matrix2f(
                dot(v[i].s, dh_du), dot(v[i].s, dh_dv),
                dot(v[i].t, dh_du), dot(v[i].t, dh_dv)
            );
        }

        // Derivative of specular constraint w.r.t. x_{i}
        if (at_endpoint_with_fixed_direction) {
            // When the 'wo' direction is fixed, the derivative here simplifies.
            dh_du = ili * (-v[i].dp_du + wi * dot(wi, v[i].dp_du));
            dh_dv = ili * (-v[i].dp_dv + wi * dot(wi, v[i].dp_dv));
        } else {
            // Standard case for fixed emitter position
            dh_du = -v[i].dp_du * (ili + ilo) + wi * (dot(wi, v[i].dp_du) * ili)
                                              + wo * (dot(wo, v[i].dp_du) * ilo);
            dh_dv = -v[i].dp_dv * (ili + ilo) + wi * (dot(wi, v[i].dp_dv) * ili)
                                              + wo * (dot(wo, v[i].dp_dv) * ilo);
        }
        dh_du -= h * dot(dh_du, h);
        dh_dv -= h * dot(dh_dv, h);
        if (eta != 1.f) {
            dh_du *= -1.f;
            dh_dv *= -1.f;
        }

        v[i].dC_dx_cur = Matrix2f(
            dot(v[i].ds_du, h) + dot(v[i].s, dh_du), dot(v[i].ds_dv, h) + dot(v[i].s, dh_dv),
            dot(v[i].dt_du, h) + dot(v[i].t, dh_du), dot(v[i].dt_dv, h) + dot(v[i].t, dh_dv)
        );

        // Derivative of specular constraint w.r.t. x_{i+1}
        if (i < k-1) {
            dh_du = ilo * (v[i+1].dp_du - wo * dot(wo, v[i+1].dp_du));
            dh_dv = ilo * (v[i+1].dp_dv - wo * dot(wo, v[i+1].dp_dv));

            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);
            if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            }

            v[i].dC_dx_next = Matrix2f(
                dot(v[i].s, dh_du), dot(v[i].s, dh_dv),
                dot(v[i].t, dh_du), dot(v[i].t, dh_dv)
            );
        }

        // Evaluate specular constraint
        Vector2f H(dot(v[i].s, h), dot(v[i].t, h));
        Vector3f n_offset = m_offset_normals[i];
        Vector2f N(n_offset[0], n_offset[1]);
        v[i].C = H - N;
    }

    if (!invert_tridiagonal_step(v)) {
        return false;
    }

    return true;
}

MTS_VARIANT typename SpecularManifoldMultiScatter<Float, Spectrum>::Mask
SpecularManifoldMultiScatter<Float, Spectrum>::compute_step_anglediff(const Point3f &x0,
                                                                      const EmitterInteraction &ei) {
    std::vector<ManifoldVertex> &v = m_current_path;
    bool success = true;
    size_t k = v.size();
    for (size_t i = 0; i < k; ++i) {
        v[i].C = Vector2f(0.f);
        v[i].dC_dx_prev = Matrix2f(0.f);
        v[i].dC_dx_cur  = Matrix2f(0.f);
        v[i].dC_dx_next = Matrix2f(0.f);

        Point3f x_prev = (i == 0)   ? x0   : v[i-1].p;
        Point3f x_next = (i == k-1) ? ei.p : v[i+1].p;
        Point3f x_cur  = v[i].p;

        bool at_endpoint_with_fixed_direction = (i == (k-1) && ei.is_directional());

        // Setup wi / wo
        Vector3f wo;
        if (at_endpoint_with_fixed_direction) {
            // Case of fixed 'wo' direction
            wo = ei.d;
        } else {
            // Standard case for fixed emitter position
            wo = x_next - x_cur;
        }
        Float ilo = norm(wo);
        if (ilo < 1e-3f) {
            return false;
        }
        ilo = rcp(ilo);
        wo *= ilo;

        Vector3f dwo_du_cur, dwo_dv_cur;
        if (at_endpoint_with_fixed_direction) {
            // Fixed 'wo' direction means its derivative must be zero
            dwo_du_cur = Vector3f(0.f);
            dwo_dv_cur = Vector3f(0.f);
        } else {
            // Standard case for fixed emitter position
            dwo_du_cur = -ilo * (v[i].dp_du - wo*dot(wo, v[i].dp_du));
            dwo_dv_cur = -ilo * (v[i].dp_dv - wo*dot(wo, v[i].dp_dv));
        }

        Vector3f wi = x_prev - x_cur;
        Float ili = norm(wi);
        if (ili < 1e-3f) {
            return false;
        }
        ili = rcp(ili);
        wi *= ili;

        Vector3f dwi_du_cur = -ili * (v[i].dp_du - wi*dot(wi, v[i].dp_du)),
                 dwi_dv_cur = -ili * (v[i].dp_dv - wi*dot(wi, v[i].dp_dv));

        // Set up constraint function and its derivatives
        bool success_i = false;

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
        Vector3f n_offset = m_offset_normals[i];
        Normal3f n = v[i].s * n_offset[0] +
                     v[i].t * n_offset[1] +
                     v[i].n * n_offset[2];
        Vector3f dn_du = v[i].ds_du * n_offset[0] +
                         v[i].dt_du * n_offset[1] +
                         v[i].dn_du * n_offset[2];
        Vector3f dn_dv = v[i].ds_dv * n_offset[0] +
                         v[i].dt_dv * n_offset[1] +
                         v[i].dn_dv * n_offset[2];

        auto [valid_i_refr_i, wio] = transform(wi, n, v[i].eta);
        if (valid_i_refr_i) {
            auto [to, po]   = SpecularManifold::sphcoords(wo);
            auto [tio, pio] = SpecularManifold::sphcoords(wio);
            v[i].C = Vector2f(to - tio, po - pio);

            Float dto_du, dpo_du, dto_dv, dpo_dv;
            Float dtio_du, dpio_du, dtio_dv, dpio_dv;

            // Derivative of specular constraint w.r.t. x_{i-1}
            if (i > 0) {
                Vector3f dwi_du_prev = ili * (v[i-1].dp_du - wi*dot(wi, v[i-1].dp_du)),
                         dwi_dv_prev = ili * (v[i-1].dp_dv - wi*dot(wi, v[i-1].dp_dv));
                // Vector3f dwo_du_prev = ilo * (v[i-1].dp_du - wo*dot(wo, v[i-1].dp_du)),  // = 0
                //          dwo_dv_prev = ilo * (v[i-1].dp_dv - wo*dot(wo, v[i-1].dp_dv));  // = 0
                auto [dwio_du_prev, dwio_dv_prev] = d_transform(wi, dwi_du_prev, dwi_dv_prev, n, Vector3f(0.f), Vector3f(0.f), v[i].eta);   // Possible optimization: specific implementation here that already knows some of these are 0.

                // std::tie(dto_du, dpo_du, dto_dv, dpo_dv)     = SpecularManifold::d_sphcoords(wo, dwo_du_prev, dwo_dv_prev);  // = 0
                std::tie(dtio_du, dpio_du, dtio_dv, dpio_dv) = SpecularManifold::d_sphcoords(wio, dwio_du_prev, dwio_dv_prev);

                v[i].dC_dx_prev(0,0) = -dtio_du;
                v[i].dC_dx_prev(1,0) = -dpio_du;
                v[i].dC_dx_prev(0,1) = -dtio_dv;
                v[i].dC_dx_prev(1,1) = -dpio_dv;
            }

            // Derivative of specular constraint w.r.t. x_{i}
            auto [dwio_du_cur, dwio_dv_cur] = d_transform(wi, dwi_du_cur, dwi_dv_cur, n, dn_du, dn_dv, v[i].eta);

            std::tie(dto_du, dpo_du, dto_dv, dpo_dv)     = SpecularManifold::d_sphcoords(wo, dwo_du_cur, dwo_dv_cur);
            std::tie(dtio_du, dpio_du, dtio_dv, dpio_dv) = SpecularManifold::d_sphcoords(wio, dwio_du_cur, dwio_dv_cur);

            v[i].dC_dx_cur(0,0) = dto_du - dtio_du;
            v[i].dC_dx_cur(1,0) = dpo_du - dpio_du;
            v[i].dC_dx_cur(0,1) = dto_dv - dtio_dv;
            v[i].dC_dx_cur(1,1) = dpo_dv - dpio_dv;

            // Derivative of specular constraint w.r.t. x_{i+1}
            if (i < k-1) {
                // Vector3f dwi_du_next = ili * (v[i+1].dp_du - wi*dot(wi, v[i+1].dp_du)),  // = 0
                //          dwi_dv_next = ili * (v[i+1].dp_dv - wi*dot(wi, v[i+1].dp_dv));  // = 0
                Vector3f dwo_du_next = ilo * (v[i+1].dp_du - wo*dot(wo, v[i+1].dp_du)),
                         dwo_dv_next = ilo * (v[i+1].dp_dv - wo*dot(wo, v[i+1].dp_dv));
                // auto [dwio_du_next, dwio_dv_next] = d_transform(wi, dwi_du_next, dwi_dv_next, n, Vector3f(0.f), Vector3f(0.f), v[i].eta); // = 0

                std::tie(dto_du, dpo_du, dto_dv, dpo_dv) = SpecularManifold::d_sphcoords(wo, dwo_du_next, dwo_dv_next);
                // std::tie(dtio_du, dpio_du, dtio_dv, dpio_dv) = SpecularManifold::d_sphcoords(wio, dwio_du_next, dwio_dv_next);   // = 0

                v[i].dC_dx_next(0,0) = dto_du;
                v[i].dC_dx_next(1,0) = dpo_du;
                v[i].dC_dx_next(0,1) = dto_dv;
                v[i].dC_dx_next(1,1) = dpo_dv;
            }

            success_i = true;
        }

        auto [valid_o_refr_o, woi] = transform(wo, n, v[i].eta);
        if (valid_o_refr_o && !success_i) {
            auto [ti, pi]   = SpecularManifold::sphcoords(wi);
            auto [toi, poi] = SpecularManifold::sphcoords(woi);
            v[i].C = Vector2f(ti - toi, pi - poi);

            Float dti_du, dpi_du, dti_dv, dpi_dv;
            Float dtoi_du, dpoi_du, dtoi_dv, dpoi_dv;

            // Derivative of specular constraint w.r.t. x_{i-1}
            if (i > 0) {
                Vector3f dwi_du_prev = ili * (v[i-1].dp_du - wi*dot(wi, v[i-1].dp_du)),
                         dwi_dv_prev = ili * (v[i-1].dp_dv - wi*dot(wi, v[i-1].dp_dv));
                // Vector3f dwo_du_prev = ilo * (v[i-1].dp_du - wo*dot(wo, v[i-1].dp_du)),  // = 0
                         // dwo_dv_prev = ilo * (v[i-1].dp_dv - wo*dot(wo, v[i-1].dp_dv));  // = 0
                // auto [dwoi_du_prev, dwoi_dv_prev] = d_transform(wo, dwo_du_prev, dwo_dv_prev, n, Vector3f(0.f), Vector3f(0.f), v[i].eta);   // = 0

                std::tie(dti_du, dpi_du, dti_dv, dpi_dv) = SpecularManifold::d_sphcoords(wi, dwi_du_prev, dwi_dv_prev);
                // std::tie(dtoi_du, dpoi_du, dtoi_dv, dpoi_dv) = SpecularManifold::d_sphcoords(woi, dwoi_du_prev, dwoi_dv_prev);   // = 0

                v[i].dC_dx_prev(0,0) = dti_du;
                v[i].dC_dx_prev(1,0) = dpi_du;
                v[i].dC_dx_prev(0,1) = dti_dv;
                v[i].dC_dx_prev(1,1) = dpi_dv;
            }


            // Derivative of specular constraint w.r.t. x_{i}
            auto [dwoi_du_cur, dwoi_dv_cur] = d_transform(wo, dwo_du_cur, dwo_dv_cur, n, dn_du, dn_dv, v[i].eta);

            std::tie(dti_du, dpi_du, dti_dv, dpi_dv)     = SpecularManifold::d_sphcoords(wi, dwi_du_cur, dwi_dv_cur);
            std::tie(dtoi_du, dpoi_du, dtoi_dv, dpoi_dv) = SpecularManifold::d_sphcoords(woi, dwoi_du_cur, dwoi_dv_cur);

            v[i].dC_dx_cur(0,0) = dti_du - dtoi_du;
            v[i].dC_dx_cur(1,0) = dpi_du - dpoi_du;
            v[i].dC_dx_cur(0,1) = dti_dv - dtoi_dv;
            v[i].dC_dx_cur(1,1) = dpi_dv - dpoi_dv;

            // Derivative of specular constraint w.r.t. x_{i+1}
            if (i < k-1) {
                // Vector3f dwi_du_next = ili * (v[i+1].dp_du - wi*dot(wi, v[i+1].dp_du)),  // = 0
                         // dwi_dv_next = ili * (v[i+1].dp_dv - wi*dot(wi, v[i+1].dp_dv));  // = 0
                Vector3f dwo_du_next = ilo * (v[i+1].dp_du - wo*dot(wo, v[i+1].dp_du)),
                         dwo_dv_next = ilo * (v[i+1].dp_dv - wo*dot(wo, v[i+1].dp_dv));
                auto [dwoi_du_next, dwoi_dv_next] = d_transform(wo, dwo_du_next, dwo_dv_next, n, Vector3f(0.f), Vector3f(0.f), v[i].eta);   // Possible optimization: specific implementation here that already knows some of these are 0.

                // std::tie(dti_du, dpi_du, dti_dv, dpi_dv)  = SpecularManifold::d_sphcoords(wi, dwi_du_next, dwi_dv_next);  // = 0
                std::tie(dtoi_du, dpoi_du, dtoi_dv, dpoi_dv) = SpecularManifold::d_sphcoords(woi, dwoi_du_next, dwoi_dv_next);

                v[i].dC_dx_next(0,0) = -dtoi_du;
                v[i].dC_dx_next(1,0) = -dpoi_du;
                v[i].dC_dx_next(0,1) = -dtoi_dv;
                v[i].dC_dx_next(1,1) = -dpoi_dv;
            }

            success_i = true;
        }

        success &= success_i;
    }

    if (!success || !invert_tridiagonal_step(v)) {
        return false;
    }
    return true;
}

MTS_VARIANT typename SpecularManifoldMultiScatter<Float, Spectrum>::Mask
SpecularManifoldMultiScatter<Float, Spectrum>::reproject(const SurfaceInteraction3f &si_) {
    m_proposed_path.clear();
    SurfaceInteraction3f si(si_);

    // Start ray-tracing towards the first specular vertex along the chain
    Point3f x0 = si.p,
            x1 = m_proposed_positions[0];
    Vector3f wo = normalize(x1 - x0);
    Ray3f ray(x0, wo, si.time, si.wavelengths);

    while (true) {
        int bounce = m_proposed_path.size();

        if (bounce >= m_config.bounces) {
            /* We reached the number of specular bounces that was requested.
               (Implicitly) connect to the light source now by terminating. */
            break;
        }

        si = m_scene->ray_intersect(ray);
        if (!si.is_valid()) {
            return false;
        }
        const ShapePtr shape = si.shape;
        if (shape != m_seed_path[bounce].shape) {
            // We hit some other shape than previously
            return false;
        }

        // Create the path vertex
        ManifoldVertex vertex = ManifoldVertex(si, 0.f);
        m_proposed_path.push_back(vertex);

        // Get current (potentially offset) normal in world space
        Vector3f n_offset = m_offset_normals[bounce];
        Vector3f m = vertex.s * n_offset[0] +
                     vertex.t * n_offset[1] +
                     vertex.n * n_offset[2];


        // Perform scattering at vertex
        Vector3f wi = -wo;
        bool scatter_success = false;
        if (vertex.eta == 1.f) {
            std::tie(scatter_success, wo) = SpecularManifold::reflect(wi, m);
        } else {
            std::tie(scatter_success, wo) = SpecularManifold::refract(wi, m, vertex.eta);
        }

        if (!scatter_success) {
            // We must have hit total internal reflection. Abort.
            return false;
        }

        ray = si.spawn_ray(wo);
    }

    return m_proposed_path.size() == m_seed_path.size();
}

MTS_VARIANT Spectrum
SpecularManifoldMultiScatter<Float, Spectrum>::specular_reflectance(const SurfaceInteraction3f &si_,
                                                                    const EmitterInteraction &ei) const {
    if (m_current_path.size() == 0) return 0.f;

    SurfaceInteraction3f si(si_);

    // Start ray-tracing towards the first specular vertex along the chain
    Point3f x0 = si.p,
            x1 = m_current_path[0].p;
    Vector3f wo = normalize(x1 - x0);
    Ray3f ray(x0, wo, si.time, si.wavelengths);

    Spectrum bsdf_val(1.f);

    for (size_t k = 0; k < m_current_path.size(); ++k) {
        si = m_scene->ray_intersect(ray);
        if (!si.is_valid()) {
            return 0.f;
        }

        // Prepare fore BSDF evaluation
        const BSDFPtr bsdf = si.bsdf();

        Vector3f wo;
        if (k < m_current_path.size() - 1) {
            // Safely connect with next vertex
            Point3f p_next = m_current_path[k+1].p;
            wo = normalize(p_next - si.p);
        } else {
            // Connect with light source
            if (ei.is_directional()) {
                wo = ei.d;
            } else {
                wo = normalize(ei.p - si.p);
            }
        }

        Complex<Spectrum> ior = si.bsdf()->ior(si);
        Mask reflection = any(neq(0.f, imag(ior)));

        Spectrum f;
        if (bsdf->roughness() > 0.f) {
            // Glossy BSDF: evaluate BSDF and transform to half-vector domain.
            BSDFContext ctx;
            Vector3f wo_l = si.to_local(wo);
            f = bsdf->eval(ctx, si, wo_l);

            /* Compared to Eq. 6 in [Hanika et al. 2015 (MNEE)], two terms are omitted:
               1) abs_dot(wo, n) is part of BSDF::eval
               2) abs_dot(h, n)  is part of the Microfacet distr. (also in BSDF::eval) */
            Vector3f h_l;
            if (reflection) {
                h_l = normalize(si.wi + wo_l);
                f *= 4.f*abs_dot(wo_l, h_l);
            } else {
                Float eta = hmean(real(ior));
                if (Frame3f::cos_theta(si.wi) < 0.f) {
                    eta = rcp(eta);
                }
                h_l = -normalize(si.wi + eta*wo_l);
                f *= sqr(dot(si.wi, h_l) + eta*dot(wo_l, h_l)) / (eta*eta * abs_dot(wo_l, h_l));
            }
        } else {
            // Delta BSDF: just account for Fresnel term and solid angle compression
            Frame3f frame = bsdf->frame(si, 0.f);
            Float cos_theta = dot(frame.n, wo);
            if (reflection) {
                if (all(eq(imag(ior), 0.f))) {
                    auto [F_, cos_theta_t, eta_it, eta_ti] = fresnel(Spectrum(abs(cos_theta)), real(ior));
                    f = F_;
                } else {
                    f = fresnel_conductor(Spectrum(abs(cos_theta)), ior);
                }
            } else {
                Float eta = hmean(real(ior));
                if (cos_theta < 0.f) {
                    eta = rcp(eta);
                }
                auto [F, unused_0, unused_1, unused_2] = fresnel(cos_theta, eta);
                f = 1.f - F;
                f *= sqr(eta);
            }

            bsdf_val *= f;
        }

        ray = Ray3f(si.p, wo, si.time, si.wavelengths);
    }

    // Do one more ray-trace towards light source to check for visibility there
    if (!ei.is_directional()) {
        ray = Ray3f(si.p, normalize(ei.p - si.p),
                    math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                    norm(ei.p - si.p) * (1.f - math::RayEpsilon<Float>),
                    si.time, si.wavelengths);
    }

    if (m_scene->ray_test(ray)) {
        return 0.f;
    }

    return bsdf_val;
}

MTS_VARIANT Float
SpecularManifoldMultiScatter<Float, Spectrum>::geometric_term(const ManifoldVertex &vx,
                                                              const ManifoldVertex &vy) {
    // First assemble full path, including all endpoints (use m_proposed_path as buffer here)
    m_proposed_path.clear();

    if (vy.fixed_direction) {
        // In this case, the resulting linear system is easier to solve if we "reverse" the path actually
        m_proposed_path.push_back(vy);
        for (int i = m_current_path.size() - 1; i >= 0; --i) {
            m_proposed_path.push_back(m_current_path[i]);
        }
        m_proposed_path.push_back(vx);
    } else {
        // Assemble path in normal "forward" ordering, leaving out start point 'x' as it's not needed
        for (size_t i = 0; i < m_current_path.size(); ++i) {
            m_proposed_path.push_back(m_current_path[i]);
        }
        m_proposed_path.push_back(vy);
    }

    // Do all the following work on this new path
    std::vector<ManifoldVertex> &v = m_proposed_path;

    size_t k = v.size();
    for (size_t i = 0; i < k-1; ++i) {
        v[i].dC_dx_prev = Matrix2f(0.f);
        v[i].dC_dx_cur  = Matrix2f(0.f);
        v[i].dC_dx_next = Matrix2f(0.f);

        Point3f x_cur  = v[i].p;
        Point3f x_next = v[i+1].p;

        Vector3f wo = x_next - x_cur;
        Float ilo = norm(wo);
        if (ilo < 1e-3f) {
            return false;
        }
        ilo = rcp(ilo);
        wo *= ilo;

        if (v[i].fixed_direction) {
            // Derivative of directional constraint w.r.t. x_{i}
            Vector3f dc_du_cur = -ilo * (v[i].dp_du - wo * dot(wo, v[i].dp_du)),
                     dc_dv_cur = -ilo * (v[i].dp_dv - wo * dot(wo, v[i].dp_dv));
            v[i].dC_dx_cur = Matrix2f(
                dot(dc_du_cur, v[i].dp_du), dot(dc_dv_cur, v[i].dp_du),
                dot(dc_du_cur, v[i].dp_dv), dot(dc_dv_cur, v[i].dp_dv)
            );

            // Derivative of directional constraint w.r.t. x_{i+1}
            Vector3f dc_du_next = ilo * (v[i+1].dp_du - wo * dot(wo, v[i+1].dp_du)),
                     dc_dv_next = ilo * (v[i+1].dp_dv - wo * dot(wo, v[i+1].dp_dv));
            v[i].dC_dx_next = Matrix2f(
                dot(dc_du_next, v[i].dp_du), dot(dc_dv_next, v[i].dp_du),
                dot(dc_du_next, v[i].dp_dv), dot(dc_dv_next, v[i].dp_dv)
            );
            continue;
        }

        Point3f x_prev = (i == 0) ? vx.p : v[i-1].p;  // Note that we only end up here for positionally fixed endpoints, thus x is not part of the path array directly.

        Vector3f wi = x_prev - x_cur;
        Float ili = norm(wi);
        if (ili < 1e-3f) {
            return false;
        }
        ili = rcp(ili);
        wi *= ili;

        // Setup generalized half-vector
        Float eta = v[i].eta;
        if (dot(wi, v[i].gn) < 0.f) {
            eta = rcp(eta);
        }
        Vector3f h = wi + eta * wo;
        if (eta != 1.f) h *= -1.f;
        Float ilh = rcp(norm(h));
        h *= ilh;

        ilo *= eta * ilh;
        ili *= ilh;

        // Local shading tangent frame
        Float dot_dpdu_n = dot(v[i].dp_du, v[i].n),
              dot_dpdv_n = dot(v[i].dp_dv, v[i].n);
        Vector3f s = v[i].dp_du - dot_dpdu_n * v[i].n,
                 t = v[i].dp_dv - dot_dpdv_n * v[i].n;

        Vector3f dh_du, dh_dv;

        // Derivative of specular constraint w.r.t. x_{i-1}
        if (i > 0) {
            dh_du = ili * (v[i-1].dp_du - wi * dot(wi, v[i-1].dp_du));
            dh_dv = ili * (v[i-1].dp_dv - wi * dot(wi, v[i-1].dp_dv));
            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);
            if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            }

            v[i].dC_dx_prev = Matrix2f(
                dot(s, dh_du), dot(s, dh_dv),
                dot(t, dh_du), dot(t, dh_dv)
            );
        }

        // Derivative of specular constraint w.r.t. x_{i}
        dh_du = -v[i].dp_du * (ili + ilo) + wi * (dot(wi, v[i].dp_du) * ili)
                                          + wo * (dot(wo, v[i].dp_du) * ilo);
        dh_dv = -v[i].dp_dv * (ili + ilo) + wi * (dot(wi, v[i].dp_dv) * ili)
                                          + wo * (dot(wo, v[i].dp_dv) * ilo);
        dh_du -= h * dot(dh_du, h);
        dh_dv -= h * dot(dh_dv, h);
        if (eta != 1.f) {
            dh_du *= -1.f;
            dh_dv *= -1.f;
        }

        Float dot_h_n    = dot(h, v[i].n),
              dot_h_dndu = dot(h, v[i].dn_du),
              dot_h_dndv = dot(h, v[i].dn_dv);

        v[i].dC_dx_cur = Matrix2f(
            dot(dh_du, s) - dot(v[i].dp_du, v[i].dn_du) * dot_h_n - dot_dpdu_n * dot_h_dndu,
            dot(dh_dv, s) - dot(v[i].dp_du, v[i].dn_dv) * dot_h_n - dot_dpdu_n * dot_h_dndv,
            dot(dh_du, t) - dot(v[i].dp_dv, v[i].dn_du) * dot_h_n - dot_dpdv_n * dot_h_dndu,
            dot(dh_dv, t) - dot(v[i].dp_dv, v[i].dn_dv) * dot_h_n - dot_dpdv_n * dot_h_dndv
        );

        // Derivative of specular constraint w.r.t. x_{i+1}
        dh_du = ilo * (v[i+1].dp_du - wo * dot(wo, v[i+1].dp_du));
        dh_dv = ilo * (v[i+1].dp_dv - wo * dot(wo, v[i+1].dp_dv));
        dh_du -= h * dot(dh_du, h);
        dh_dv -= h * dot(dh_dv, h);
        if (eta != 1.f) {
            dh_du *= -1.f;
            dh_dv *= -1.f;
        }

        v[i].dC_dx_next = Matrix2f(
            dot(s, dh_du), dot(s, dh_dv),
            dot(t, dh_du), dot(t, dh_dv)
        );
    }

    if (vy.fixed_direction) {
        Float G = invert_tridiagonal_geo(v);
        // Cancel out cosine term that will be added during lighting integral in integrator
        Vector3f d = normalize(v[k-1].p - v[k-2].p);
        G /= abs_dot(d, v[k-1].n);
        return G;
    } else {
        Float dx1_dxend = invert_tridiagonal_geo(v);
        /* Unfortunately, these geometric terms can be unstable, so to avoid
           severe variance we need to clamp here. */
        dx1_dxend = min(dx1_dxend, Float(2.f));
        Vector3f d = vx.p - v[0].p;
        Float inv_r2 = rcp(squared_norm(d));
        d *= sqrt(inv_r2);
        Float dw0_dx1 = abs_dot(d, v[0].gn) * inv_r2;
        Float G = dw0_dx1 * dx1_dxend;
        return G;
    }
}

MTS_VARIANT typename SpecularManifoldMultiScatter<Float, Spectrum>::Mask
SpecularManifoldMultiScatter<Float, Spectrum>::invert_tridiagonal_step(std::vector<ManifoldVertex> &v) {
    // Solve block tri-diagonal linear system with full RHS vector

    // From "The Natural-Constraint Representation of the Path Space for Efficient Light Transport Simulation"
    // by Kaplanyan et al. 2014 Supplemental material, Figure 2.

    auto invert = [](const Matrix2f &A, Matrix2f &Ainv) {
        Float determinant = det(A);
        if (abs(determinant) == 0) {
            return false;
        }
        Ainv = inverse(A);
        return true;
    };

    int n = int(v.size());
    if (n == 0) return true;

    v[0].tmp = v[0].dC_dx_prev;
    Matrix2f m = v[0].dC_dx_cur;
    if (!invert(m, v[0].inv_lambda)) return false;

    for (int i = 1; i < n; ++i) {
        v[i].tmp = v[i].dC_dx_prev * v[i-1].inv_lambda;
        Matrix2f m = v[i].dC_dx_cur - v[i].tmp * v[i-1].dC_dx_next;
        if (!invert(m, v[i].inv_lambda)) return false;
    }

    v[0].dx = v[0].C;
    for (int i = 1; i < n; ++i) {
        v[i].dx = v[i].C - v[i].tmp * v[i-1].dx;
    }

    v[n-1].dx = v[n-1].inv_lambda * v[n-1].dx;
    for (int i = n-2; i >= 0; --i) {
        v[i].dx = v[i].inv_lambda * (v[i].dx - v[i].dC_dx_next * v[i+1].dx);
    }

    return true;
}

MTS_VARIANT Float
SpecularManifoldMultiScatter<Float, Spectrum>::invert_tridiagonal_geo(std::vector<ManifoldVertex> &v) {
    // Solve block tri-diagonal linear system with RHS vector where only last element in non-zero

    // Procedure as outlined in original "Manifold Exploration" by Jakob and Marschner 2012.
    // Based on the implementation in Mitsuba 0.6: manifold.cpp line 382

    auto invert = [](const Matrix2f &A, Matrix2f &Ainv) {
        Float determinant = det(A);
        if (abs(determinant) == 0) {
            return false;
        }
        Ainv = inverse(A);
        return true;
    };

    int n = int(v.size());
    if (n == 0) return 0.f;

    Matrix2f Li;
    if (!invert(v[0].dC_dx_cur, Li)) return 0.f;

    for (int i = 0; i < n-2; ++i) {
        v[i].tmp = Li * v[i].dC_dx_next;
        Matrix2f m = v[i+1].dC_dx_cur - v[i+1].dC_dx_prev * v[i].tmp;
        if (!invert(m, Li)) return 0.f;
    }

    v[n-2].inv_lambda = -Li * v[n-2].dC_dx_next;
    for (int i = n-3; i >= 0; --i) {
        v[i].inv_lambda = -v[i].tmp * v[i+1].inv_lambda;
    }

    return abs(det(-v[0].inv_lambda));
}

MTS_VARIANT void SpecularManifoldMultiScatter<Float, Spectrum>::print_statistics() {
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

MTS_INSTANTIATE_CLASS(SpecularManifoldMultiScatter)
NAMESPACE_END(mitsuba)

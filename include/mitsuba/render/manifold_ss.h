#pragma once

#include <mitsuba/render/manifold.h>

NAMESPACE_BEGIN(mitsuba)

/// Datastructure handling specular manifold sampling in the one-bounce case.
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER SpecularManifoldSingleScatter {
public:
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape)
    using BSDFPtr            = typename RenderAliases::BSDFPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using ManifoldVertex     = ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;
    using SpecularManifold   = SpecularManifold<Float, Spectrum>;

    /// Initialize data structure
    SpecularManifoldSingleScatter(const Scene *scene, const SMSConfig &config);
    ~SpecularManifoldSingleScatter();

    // ========================================================================
    //           Main functionality, to be called from integrators
    // ========================================================================

    /**
     * \brief Perform specular manifold sampling, with parameters based on the
     * current configuration.
     *
     *
     * \param si
     *     Current shading point interaction.
     *
     * \param sampler
     *     Reference to the sampler to use for RNG
     *
     * \return
     *     An estimate of the (single-bounce) caustic path contribution at
     *     the desired shading point.
     */
    Spectrum specular_manifold_sampling(const SurfaceInteraction3f &si,
                                        ref<Sampler> sampler) const;


    // ========================================================================
    //           Helper functions for internal use, or debugging
    // ========================================================================

    /**
     * \brief Sample one path via specular manifold sampling. The (inverse) pdf
     * of this process can be estimated (either in unbiased or biased way) by
     * repeatedly calling this method, compare with Algorithm 2 in the paper.
     *
     * \param shape
     *     Specular shape to start to use in the path.
     *
     * \param si
     *     Current shading point interaction. This will be the start of the
     *     sampled path segment.
     *
     * \param ei
     *     Sampled emitter interaction
     *
     * \param sampler
     *     Reference to the sampler to use for RNG
     *
     * \param n_offset
     *     (Optional) offset normal to use for during the Newton solver steps.
     *     E.g. sampled from a microfacet distribution. Should be in standard
     *     local coordinate system (Z-axis = up).
     *     This is relevant only for rough/glossy events. (Default: [0, 0, 1])
     *
     * p_start: (Optionally) override the sampled initial position on the shape
     *          (Only useful for debugging or visualization purposes.)
     *
     * \return A tuple (success, si_final, si_initial) consisting of
     *
     *     success: Did the sampling produce a solution?
     *
     *     si_final: The resulting surface interaction (in case of success)
     *
     *     si_initial: The initial surface interaction, produced by uniformly
     *                 sampling on \ref shape. (Only useful for debugging or
     *                 visualization purposes.)
     */
    std::tuple<Mask, SurfaceInteraction3f, SurfaceInteraction3f>
    sample_path(const ShapePtr shape,
                const SurfaceInteraction3f &si,
                const EmitterInteraction &ei,
                ref<Sampler> sampler,
                const Vector3f &n_offset = Vector3f(0.f, 0.f, 1.f),
                const Point3f &p_start = Point3f(0.f)) const;

    /**
     * \brief Evaluate throughput for a sampled path segment. Does not account
     * for the (inverse) probability of sampling the path, which needs to be
     * estimated separately bu repeatedly calling 'sample_path'.
     *
     * \param si
     *     Current shading point interaction.
     *
     * \param ei
     *     Sampled emitter interaction
     *
     * \param si_final
     *     Specular solution / result from SMS
     *
     * \return
     *     Final contribution, involving generalized geometric term, reflectance
     *     at the specular event, and emitter weight.
     */
    Spectrum evaluate_path_contribution(const SurfaceInteraction3f &si, const EmitterInteraction &ei,
                                        const SurfaceInteraction3f &si_final) const;


    /// Newton solver to find admissable path segment
    std::pair<Mask, SurfaceInteraction3f>
    newton_solver(const SurfaceInteraction3f &si,
                  const ManifoldVertex &vtx_init,
                  const EmitterInteraction &ei,
                  const Vector3f &n_offset = Vector3f(0.f, 0.f, 1.f),
                  Float smoothing = 0.f) const;

    /// Evaluate constraint and tangent step in the half-vector formulation
    std::tuple<Mask, Vector2f, Vector2f>
    compute_step_halfvector(const Point3f &v0p,
                            const ManifoldVertex &v1,
                            const EmitterInteraction &v2,
                            const Vector3f &n_offset = Vector3f(0.f, 0.f, 1.f)) const;

    /// Evaluate constraint and tangent step in the angle difference formulation
    std::tuple<Mask, Vector2f, Vector2f>
    compute_step_anglediff(const Point3f &v0p,
                           const ManifoldVertex &v1,
                           const EmitterInteraction &v2,
                           const Vector3f &n_offset = Vector3f(0.f, 0.f, 1.f)) const;

    /// Evalaute reflectance at specular interaction towards light source
    Spectrum specular_reflectance(const SurfaceInteraction3f &si,
                                  const Vector3f &wo) const;

    /// Compute generlized geometric term between v0 and v2, via specular vertex v1
    Float geometric_term(const ManifoldVertex &v0,
                         const ManifoldVertex &v1,
                         const ManifoldVertex &v2) const;

    static void print_statistics();

protected:
    static std::atomic<int> stats_solver_failed;
    static std::atomic<int> stats_solver_succeeded;
    static std::atomic<int> stats_bernoulli_trial_calls;
    static std::atomic<int> stats_bernoulli_trial_iterations;
    static std::atomic<int> stats_bernoulli_trial_iterations_max;

protected:
    const Scene *m_scene = nullptr;
    SMSConfig m_config;
};

NAMESPACE_END(mitsuba)

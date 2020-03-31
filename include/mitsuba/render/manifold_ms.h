#pragma once

#include <mitsuba/render/manifold.h>

NAMESPACE_BEGIN(mitsuba)

/// Datastructure handling specular manifold sampling in the multi-bounce case.
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER SpecularManifoldMultiScatter /* : public Object */ {
public:
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape)
    using BSDFPtr            = typename RenderAliases::BSDFPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using ManifoldVertex     = ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;
    using SpecularManifold   = SpecularManifold<Float, Spectrum>;

    /// Initialize data structure
    SpecularManifoldMultiScatter() {}
    ~SpecularManifoldMultiScatter() {}

    // ========================================================================
    //           Main functionality, to be called from integrators
    // ========================================================================}

    void init(const Scene *scene, const SMSConfig &config);

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
                                        ref<Sampler> sampler);


    // ========================================================================
    //           Helper functions for internal use, or debugging
    // ========================================================================

    /**
     * \brief Sample one path via specular manifold sampling. The (inverse) pdf
     * of this process can be estimated (either in unbiased or biased way) by
     * repeatedly calling this method, compare with Algorithm 2 in the paper.
     *
     * Compared to the one-bounce implementation, there is currently no support
     * for the "two-stage" solver for normal maps (though it would be compatible
     * in principle).
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
     * \param first_path
     *     Is this the first path of one estimate? If so we need to also
     *     sample offset normals whenever we hit a glossy object.
     *
     * p_start: (Optionally) override the sampled initial position on the shape
     *          (Only useful for debugging or visualization purposes.)
     *
     * \return
     *     Did the sampling succeed?
     */
    Mask sample_path(const ShapePtr shape,
                     const SurfaceInteraction3f &si,
                     const EmitterInteraction &ei,
                     ref<Sampler> sampler,
                     bool first_path,
                     const Point3f &p_start = Point3f(0.f));

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
     * \return
     *     Final contribution, involving generalized geometric term, reflectance
     *     at the specular event, emitter weight, and visibility.
     */
    Spectrum evaluate_path_contribution(const SurfaceInteraction3f &si,
                                        const EmitterInteraction &ei);

    /// Sample initial seed path, to be corrected with the Newton solver
    Mask sample_seed_path(const ShapePtr shape,
                          const SurfaceInteraction3f &si,
                          const EmitterInteraction &ei,
                          ref<Sampler> sampler,
                          bool first_path,
                          const Point3f &p_start = Point3f(0.f));

    /// Newton solver to find admissable path segment
    Mask newton_solver(const SurfaceInteraction3f &si,
                       const EmitterInteraction &ei);

    /// Evaluate constraint and tangent step in the half-vector formulation
    Mask compute_step_halfvector(const Point3f &v0p,
                                 const EmitterInteraction &v2);

    /// Evaluate constraint and tangent step in the angle difference formulation
    Mask compute_step_anglediff(const Point3f &v0p,
                                const EmitterInteraction &v2);

    /// Reproject proposed offset positions back to surfaces
    Mask reproject(const SurfaceInteraction3f &si);

    /// Evalaute reflectance at specular interaction towards light source
    Spectrum specular_reflectance(const SurfaceInteraction3f &si,
                                  const EmitterInteraction &ei) const;

    /// Compute generlized geometric term between vx and vy, via multiple specular vertices
    Float geometric_term(const ManifoldVertex &vx,
                         const ManifoldVertex &vy);

    /// From the built-up tridiagonal block matrix, compute the steps
    Mask invert_tridiagonal_step(std::vector<ManifoldVertex> &v);

    /// From the built-up tridiagonal block matrix, compute the geometric term
    Float invert_tridiagonal_geo(std::vector<ManifoldVertex> &v);

    const std::vector<ManifoldVertex> &current_path() const { return m_current_path; }

    static void print_statistics();

protected:
    static std::atomic<int> stats_solver_failed;
    static std::atomic<int> stats_solver_succeeded;
    static std::atomic<int> stats_bernoulli_trial_calls;
    static std::atomic<int> stats_bernoulli_trial_iterations;
    static std::atomic<int> stats_bernoulli_trial_iterations_max;

    // MTS_DECLARE_CLASS()
protected:
    const Scene *m_scene = nullptr;
    SMSConfig m_config;

    std::vector<ManifoldVertex> m_seed_path, m_current_path, m_proposed_path;
    std::vector<Point3f> m_proposed_positions;

    std::vector<Vector3f> m_offset_normals;
    Float m_offset_normals_pdf;
};

NAMESPACE_END(mitsuba)

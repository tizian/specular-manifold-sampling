#pragma once

#include <mitsuba/render/manifold.h>

NAMESPACE_BEGIN(mitsuba)

/// Datastructure handling specular manifold sampling in the multi-bounce case.
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER SpecularManifoldGlints {
public:
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape)
    using BSDFPtr            = typename RenderAliases::BSDFPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using EmitterPtr         = typename RenderAliases::EmitterPtr;
    using SensorPtr          = typename RenderAliases::SensorPtr;
    using ManifoldVertex     = ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;
    using SpecularManifold   = SpecularManifold<Float, Spectrum>;

    /// Initialize data structure
    SpecularManifoldGlints(const Scene *scene, const SMSConfig &config);

    virtual ~SpecularManifoldGlints();

    // ========================================================================
    //           Main functionality, to be called from integrators
    // ========================================================================

    /**
     * \brief Perform specular manifold sampling for glints, with parameters
     * based on the current configuration.
     *
     * Internally performs MIS between SMS sampling and BSDF sampling on a
     * high-frequency normal mapped material.
     *
     * \param sensor_position
     *     Point on the camera aperture / primary ray origin
     *
     * \param si
     *     Current (glinty) shading point interaction.
     *
     * \param sampler
     *     Reference to the sampler to use for RNG
     *
     * \return A tuple (contribution, bsdf_weight, bsdf_wo) consisting of
     *
     *      contribution: Estimate of the glinty contribution at the
     *                    shading point which is the result of MIS
     *                    between SMS and BSDF sampling strategies.
     *
     *      bsdf_weight: BSDF sampling weight produce
     */
    Spectrum specular_manifold_sampling(const Point3f &sensor_position,
                                        const SurfaceInteraction3f &si,
                                        ref<Sampler> sampler) const;

    // ========================================================================
    //           Helper functions for internal use, or debugging
    // ========================================================================

    /**
     * \brief Sample "glint" contribution from a specular surface
     *
     * \param sensor_position
     *     Point on the camera aperture / primary ray origin
     *
     * \param ei
     *     Sampled emitter interaction
     *
     * \param si
     *     Shading point interaction of the glinty surface
     *
     * \param sampler
     *     Reference to sampler to use for RNG
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
     * \return A tuple (success, uv_final, uv_initial) consisting of
     *
     *     success: Did the sampling produce a solution?
     *
     *     uv_final: The resulting UV position (in case of success)
     *
     *     uv_initial: The initial UV position, produced by uniformly sampling
     *                 inside the pixel footprint. (Only useful for debugging or
     *                 visualization purposes.)
     */
    std::tuple<Mask, Point2f, Point2f>
    sample_glint(const Point3f &sensor_position,
                 const EmitterInteraction &ei,
                 const SurfaceInteraction3f &si,
                 ref<Sampler> sampler,
                 const Vector3f &n_offset = Vector3f(0.f, 0.f, 1.f),
                 const Point2f xi_start = Point2f(1.f)) const;

    /**
     * \brief Evaluate throughput for a sampled glint position. Does not account
     * for the (inverse) probability of sampling the glint, which needs to be
     * estimated separately by repeatedly calling 'sample_path'.
     *
     * \param sensor_position
     *     Point on the camera aperture / primary ray origin
     *
     * \param ei
     *     Sampled emitter interaction
     *
     * \param si
     *     Current shading point interaction.
     *
     * \param uv
     *     UV position of the glint
     *
     * \return
     *     Final contribution, involving generalized geometric term, reflectance
     *     at the specular event, and emitter weight.
     */
    Spectrum evaluate_glint_contribution(const Point3f &sensor_position,
                                         const EmitterInteraction &ei,
                                         const SurfaceInteraction3f &si,
                                         const Point2f &uv) const;

    /// Newton solver to find admissable glint position in UV space
    std::pair<Mask, Point2f>
    newton_solver(const Point2f &target_slope,
                  const Point2f &uv_init,
                  const SurfaceInteraction3f &si) const;

    /// Evaluate constraint function and compute the next step
    std::tuple<Mask, Point2f, Vector2f>
    compute_step(const Point2f &target_slope,
                 const Point2f &uv,
                 const SurfaceInteraction3f &si) const;

    /// Evalaute reflectance at specular interaction towards light source
    Spectrum specular_reflectance(const SurfaceInteraction3f &si,
                                  const Vector3f &wo) const;

    /// Compute generlized geometric term between v0 and v2, via specular vertex v1
    Float geometric_term(const ManifoldVertex &v0,
                         const ManifoldVertex &v1,
                         const ManifoldVertex &v2) const;

    /// LEAN mapping directional sampling density used for glint MIS
    Float lean_pdf(const SurfaceInteraction3f &si,
                   const Vector3f &wo) const;

    /// Multiple importance sampling power heuristic
    MTS_INLINE Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    /// Test if point lies in triangle
    template <typename Point2,
              typename Value   = value_t<Point2>,
              typename Mask    = mask_t<Value>,
              typename Vector2 = Vector<Value, 2>>
    MTS_INLINE
    Mask inside_triangle(const Point2 &p,
                         const Point2f &a, const Point2f &b, const Point2f &c) const {
        Vector2 v0 = c - a,
                v1 = b - a,
                v2 = p - a;

        Value dot00 = dot(v0, v0),
              dot01 = dot(v0, v1),
              dot02 = dot(v0, v2),
              dot11 = dot(v1, v1),
              dot12 = dot(v1, v2);

        Value inv_denom = rcp(dot00 * dot11 - dot01 * dot01);
        Value u = (dot11 * dot02 - dot01 * dot12) * inv_denom,
              v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
        return (u >= 0) & (v >= 0) & (u + v < 1);
    }

    /// Test if point lies in parallelogram
    template <typename Point2,
              typename Value   = value_t<Point2>,
              typename Mask    = mask_t<Value>>
    MTS_INLINE
    Mask inside_parallelogram(const Point2 &p,
                              const Point2f &a, const Point2f &b, const Point2f &c, const Point2f &d) const {
        return inside_triangle(p, a, b, c) | inside_triangle(p, a, c, d);
    }

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

#pragma once

#include <mitsuba/render/manifold.h>
#include <mitsuba/core/random.h>

NAMESPACE_BEGIN(mitsuba)

/* Added after submission: a slightly simplified version of the SMS variant
   for glints that additionally makes use of SIMD vectorization on modern CPU
   architectures. This way, e.g. on AVX512, we can run 16 instances of the
   Newton solver at the same time to find multiple glints.

   Compared to the scalar implementation (manifold_glints.h), the vectorized
   variant only works on purely specular materials (without surface roughness)
   and only the biased pdf estimate is supported. Both could be added with some
   additional API changes.*/

#define VECTOR_PACKET_SIZE 16   // How many glints should be sampled at once with vectorization?

/// Datastructure handling specular manifold sampling in the glints case.
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER SpecularManifoldGlintsVectorized {
public:
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape)
    using BSDFPtr            = typename RenderAliases::BSDFPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using EmitterPtr         = typename RenderAliases::EmitterPtr;
    using SensorPtr          = typename RenderAliases::SensorPtr;
    using ManifoldVertex     = mitsuba::ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;
    using SpecularManifold   = SpecularManifold<Float, Spectrum>;

    using FloatP          = Array<Float, VECTOR_PACKET_SIZE>;
    using UInt32P         = uint32_array_t<FloatP>;
    using PCG32           = mitsuba::PCG32<UInt32P>;
    using MaskP           = mask_t<FloatP>;
    using Point2fP        = Point<FloatP, 2>;
    using Point2uP        = Point<UInt32P, 2>;
    using Vector2fP       = Vector<FloatP, 2>;
    using Vector3fP       = Vector<FloatP, 3>;
    using Matrix2fP       = Matrix<FloatP, 2>;
    using SpectrumP       = replace_scalar_t<FloatP, Spectrum>;
    using ManifoldVertexP = mitsuba::ManifoldVertex<FloatP, Spectrum>;

    /// Initialize data structure
    SpecularManifoldGlintsVectorized() {}
    ~SpecularManifoldGlintsVectorized() {}

    // ========================================================================
    //           Main functionality, to be called from integrators
    // ========================================================================

    void init_sampler(ref<Sampler> sampler);

    void init(const Scene *scene, const SMSConfig &config);

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
     * \return A pair (success, uv_final) consisting of
     *
     *     success: Did the sampling produce a solution?
     *
     *     uv_final: The resulting UV position (in case of success)
     */
    std::pair<MaskP, Point2fP>
    sample_glint(const Point3f &sensor_position,
                 const EmitterInteraction &ei,
                 const SurfaceInteraction3f &si) const;

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
    SpectrumP evaluate_glint_contribution(const Point3f &sensor_position,
                                          const EmitterInteraction &ei,
                                          const SurfaceInteraction3f &si,
                                          const Point2fP &uv,
                                          MaskP active) const;

    /// Newton solver to find admissable glint position in UV space
    std::pair<MaskP, Point2fP>
    newton_solver(const Point2f &target_slope,
                  const Point2fP &uv_init,
                  const SurfaceInteraction3f &si) const;

    /// Evaluate constraint function and compute the next step
    std::tuple<MaskP, Point2fP, Vector2fP>
    compute_step(const Point2f &target_slope,
                 const Point2fP &uv,
                 const SurfaceInteraction3f &si,
                 MaskP active) const;

    /// Evalaute reflectance at specular interaction towards light source
    SpectrumP specular_reflectance(const SurfaceInteraction3f &si,
                                   const Vector3f &wo,
                                   const Point2fP &uv,
                                   MaskP active) const;

    /// Compute generlized geometric term between v0 and v2, via specular vertex v1
    template <typename ManifoldVertexT,
              typename FloatT = typename ManifoldVertexT::Float,
              typename MaskT  = mask_t<FloatT>>
    FloatT geometric_term(const ManifoldVertex &v0,
                          const ManifoldVertexT &v1,
                          const ManifoldVertex &v2,
                          MaskT active) const {
        using Vector3fT = Vector<FloatT, 3>;
        using Matrix2fT = Matrix<FloatT, 2>;

        Vector3fT wi = v0.p - v1.p;
        FloatT ili = norm(wi);
        active &= ili >= 1e-3f;
        ili = rcp(ili);
        wi *= ili;

        Vector3fT wo = v2.p - v1.p;
        FloatT ilo = norm(wo);
        active &= ilo >= 1e-3f;
        ilo = rcp(ilo);
        wo *= ilo;

        Matrix2fT dc1_dx0, dc2_dx1, dc2_dx2;
        if (v2.fixed_direction) {
            /* This case is actually a bit more tricky as we're now in a situation
               with two "specular" constraints. As a consequence, we need to solve
               a bigger matrix system, so we prepare a few additional terms. */

            // Derivative of directional light constraint w.r.t. v1
            Vector3fT dc2_du1 = ilo * (v1.dp_du - wo * dot(wo, v1.dp_du)),
                      dc2_dv1 = ilo * (v1.dp_dv - wo * dot(wo, v1.dp_dv));
            dc2_dx1 = Matrix2fT(
                dot(dc2_du1, v2.dp_du), dot(dc2_dv1, v2.dp_du),
                dot(dc2_du1, v2.dp_dv), dot(dc2_dv1, v2.dp_dv)
            );

            // Derivative of directional light constraint w.r.t. v2
            Vector3fT dc2_du2 = -ilo * (v2.dp_du - wo * dot(wo, v2.dp_du)),
                      dc2_dv2 = -ilo * (v2.dp_dv - wo * dot(wo, v2.dp_dv));
            dc2_dx2 = Matrix2fT(
                dot(dc2_du2, v2.dp_du), dot(dc2_dv2, v2.dp_du),
                dot(dc2_du2, v2.dp_dv), dot(dc2_dv2, v2.dp_dv)
            );
        }

        // Setup generalized half-vector
        Vector3fT h = wi + wo;
        FloatT ilh = rcp(norm(h));
        h *= ilh;

        ilo *= ilh;
        ili *= ilh;

        // Local shading tangent frame
        FloatT dot_dpdu_n = dot(v1.dp_du, v1.n),
               dot_dpdv_n = dot(v1.dp_dv, v1.n);
        Vector3fT s = v1.dp_du - dot_dpdu_n * v1.n,
                  t = v1.dp_dv - dot_dpdv_n * v1.n;

        Vector3fT dh_du, dh_dv;

        if (v2.fixed_direction) {
            // Derivative of specular constraint w.r.t. v0
            dh_du = ili * (v0.dp_du - wi * dot(wi, v0.dp_du));
            dh_dv = ili * (v0.dp_dv - wi * dot(wi, v0.dp_dv));
            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);
            dc1_dx0 = Matrix2fT(
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

        FloatT dot_h_n    = dot(h, v1.n),
               dot_h_dndu = dot(h, v1.dn_du),
               dot_h_dndv = dot(h, v1.dn_dv);
        Matrix2fT dc1_dx1(
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
        Matrix2fT dc1_dx2(
            dot(dh_du, s), dot(dh_dv, s),
            dot(dh_du, t), dot(dh_dv, t)
        );

        FloatT G = 0.f;
        if (v2.fixed_direction) {
            // Invert 2x2 block matrix system
            FloatT determinant = det(dc2_dx2);
            active &= abs(determinant) >= 1e-6f;

            Matrix2fT Li = inverse(dc2_dx2);
            Matrix2fT tmp = Li * dc2_dx1;
            Matrix2fT m = dc1_dx1 - dc1_dx2 * tmp;
            determinant = det(m);
            active &= abs(determinant) >= 1e-6f;

            Li = inverse(m);
            Matrix2fT sol1 = -Li * dc1_dx0;
            Matrix2fT sol0 = -tmp * sol1;

            G = abs(det(-sol0));
        } else {
            // Invert single 2x2 matrix
            FloatT determinant = det(dc1_dx1);
            active &= abs(determinant) >= 1e-6f;

            Matrix2fT inv_dc1_dx1 = inverse(dc1_dx1);
            FloatT dx1_dx2 = abs(det(inv_dc1_dx1 * dc1_dx2));

            Vector3fT d = v0.p - v1.p;
            FloatT inv_r2 = rcp(squared_norm(d));
            d *= sqrt(inv_r2);
            FloatT dw0_dx1 = abs_dot(d, v1.gn) * inv_r2;
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
        return select(active, G, 0.f);
    }

    /// Vectorized access into slopes
    Point2fP slope(const BSDFPtr bsdf,
                   const Point2fP &uv,
                   MaskP active) const;

    /// Vectorized access into slopes
    std::pair<Vector2fP, Vector2fP>
    slope_derivative(const BSDFPtr bsdf,
                     const Point2fP &uv,
                     MaskP active) const;

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

protected:
    const Scene *m_scene = nullptr;
    SMSConfig m_config;

    // Need a RNG that can generate vectorized samples (e.g. 16 x Point2f at a time for AVX512)
    std::unique_ptr<PCG32> m_rng;
};

NAMESPACE_END(mitsuba)

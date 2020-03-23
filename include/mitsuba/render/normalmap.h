#pragma once

#include <mitsuba/core/bitmap.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * Datastructure for (mipmapped) normal or LEAN map data
 *
 * We assume square textures with power-of-two resolution.
 * Normal data can either be accessed based on interpolated normals or slopes.
 */
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER Normalmap : public Object {
public:
    MTS_IMPORT_TYPES(ReconstructionFilter)

    // Optionally switch to finite differences to compute the texture derivatives
    // #define NORMAL_MAP_FINITE_DIFFERENCES

    // =============================================================
    //! @{ \name Initialization and general information
    // =============================================================

    /// Construct hierarchical data structure from normal map file
    Normalmap(const std::string &filename);
    virtual ~Normalmap();

    /// Filename of underlying normal map
    inline std::string filename() const { return m_filename; }

    /// Number of levels in the mipmap
    inline int levels() const { return m_normals_mipmap.size(); }

    /// Spatial resolution of normalmap at given mipmap level
    inline size_t resolution(size_t level) const {
        return m_sizes[std::min(level, m_normals_mipmap.size()-1)];
    }

    /**
     * \brief Evaluate normal map
     *
     * Returns the (bilinearly) interpolated RGB value stored in the underlying
     * normal map texture.
     *
     * Alternatively, in the "use_slopes" case, it returns the (bilinearly)
     * interpolated slope. This is useful to make normal mapping more
     * consistent with slope-based approaches such as LEAN mapping.
     *
     * \param mip_level
     *     Mipmap level that should be used for the lookup
     *
     * \param uv
     *     Query position along texture coordinate parameterization
     *
     * \return Either surface normal encoded as RGB value or slope, see above.
     */
    MTS_INLINE Vector3f
    eval_normal(const Point2f &uv, size_t mip_level, bool use_slopes, Mask active=true) const {
        size_t res = resolution(mip_level);
        Float duv = rcp(Float(res));

        Point2f pos = (uv - 0.5f*duv)*res;
        pos = clamp(pos, Point2f(0), Point2f(res-1));
        Point2u p_min = Point2u(floor(pos)),
                p_max = p_min + 1;
        p_min = clamp(p_min, 0, res-1);
        p_max = clamp(p_max, 0, res-1);

        UInt32 idx00 = p_min.y()*res + p_min.x(),
               idx10 = p_min.y()*res + p_max.x(),
               idx01 = p_max.y()*res + p_min.x(),
               idx11 = p_max.y()*res + p_max.x();

        Point2f w1 = pos - p_min,
                w0 = 1.f - w1;

        if (use_slopes) {
            using Vector5f = Vector<Float, 5>;
            const ScalarFloat *ptr = (const ScalarFloat *) m_lean_mipmap[mip_level]->data();
            Vector5f v00 = gather<Vector5f, 0, true>(ptr, idx00, active),
                     v10 = gather<Vector5f, 0, true>(ptr, idx10, active),
                     v01 = gather<Vector5f, 0, true>(ptr, idx01, active),
                     v11 = gather<Vector5f, 0, true>(ptr, idx11, active);
            Vector5f v0 = fmadd(w0.x(), v00, w1.x() * v10),
                     v1 = fmadd(w0.x(), v01, w1.x() * v11);
            Vector5f v = fmadd(w0.y(), v0, w1.y() * v1);
            return Vector3f(-v[0], -v[1], 1.f);
        } else {
            const ScalarFloat *ptr = (const ScalarFloat *) m_normals_mipmap[mip_level]->data();
            Vector3f v00 = gather<Vector3f, 0, true>(ptr, idx00, active),
                     v10 = gather<Vector3f, 0, true>(ptr, idx10, active),
                     v01 = gather<Vector3f, 0, true>(ptr, idx01, active),
                     v11 = gather<Vector3f, 0, true>(ptr, idx11, active);
            Vector3f v0 = fmadd(w0.x(), v00, w1.x() * v10),
                     v1 = fmadd(w0.x(), v01, w1.x() * v11);
            Vector3f v = fmadd(w0.y(), v0, w1.y() * v1);
            return v;
        }
    }

    /**
     * \brief Evaluate normal map derivatives
     *
     * Returns the (bilinearly) interpolated RGB derivatives based on the
     * underlying normal map texture.
     *
     * Alternatively, in the "use_slopes" case, it returns the (bilinearly)
     * interpolated slope derivatives. This is useful to make normal mapping
     * more consistent with slope-based approaches such as LEAN mapping.
     *
     * \param mip_level
     *     Mipmap level that should be used for the lookup
     *
     * \param uv
     *     Query position along texture coordinate parameterization
     *
     * \return Either surface normal derivatives encoded as RGB value or slope
     *         derivatives, see above.
     */
    MTS_INLINE std::pair<Vector3f, Vector3f>
    eval_normal_derivatives(const Point2f &uv, size_t mip_level, bool use_slopes, Mask active=true) const {

#ifdef NORMAL_MAP_FINITE_DIFFERENCES
        Float eps = rcp(Float(resolution(mip_level)));

        Point2f uv_u = uv + Point2f(eps, 0),
                uv_v = uv + Point2f(0, eps);

        Vector3f n, n_u, n_v;
        if (use_slopes) {
            n = eval_normal<true>(mip_level, uv_, active),
            n_u = eval_normal<true>(mip_level, uv_u, active),
            n_v = eval_normal<true>(mip_level, uv_v, active);
        } else {
            n = eval_normal<false>(mip_level, uv_, active),
            n_u = eval_normal<false>(mip_level, uv_u, active),
            n_v = eval_normal<false>(mip_level, uv_v, active);
        }

        Vector3f du = (n_u - n) / eps,
                 dv = (n_v - n) / eps;
        return std::make_pair(du, dv);
#else
        size_t res = resolution(mip_level);
        Float duv = rcp(Float(res));

        Point2f pos = (uv - 0.5f*duv)*res;
        pos = clamp(pos, Point2f(0), Point2f(res-1));
        Point2u p_min = Point2u(floor(pos)),
                p_max = p_min + 1;
        p_min = clamp(p_min, 0, res-1);
        p_max = clamp(p_max, 0, res-1);

        UInt32 idx00 = p_min.y()*res + p_min.x(),
               idx10 = p_min.y()*res + p_max.x(),
               idx01 = p_max.y()*res + p_min.x(),
               idx11 = p_max.y()*res + p_max.x();

        Point2f w = pos - p_min;

        Vector3f du, dv;
        if (use_slopes) {
            using Vector5f = Vector<Float, 5>;

            const ScalarFloat *ptr = (const ScalarFloat *) m_lean_mipmap[mip_level]->data();
            Vector5f v00 = gather<Vector5f, 0, true>(ptr, idx00, active),
                     v10 = gather<Vector5f, 0, true>(ptr, idx10, active),
                     v01 = gather<Vector5f, 0, true>(ptr, idx01, active),
                     v11 = gather<Vector5f, 0, true>(ptr, idx11, active);
            Vector5f tmp = v01 + v10 - v11;
            Vector5f tmp_u = (v10 + v00*(w[1] - 1.f) - tmp*w[1]) * Float(res),
                     tmp_v = (v01 + v00*(w[0] - 1.f) - tmp*w[0]) * Float(res);

            du = Vector3f(-tmp_u[0], -tmp_u[1], 1.f);
            dv = Vector3f(-tmp_v[0], -tmp_v[1], 1.f);
        } else {
            const ScalarFloat *ptr = (const ScalarFloat *) m_normals_mipmap[mip_level]->data();
            Vector3f v00 = gather<Vector3f, 0, true>(ptr, idx00, active),
                     v10 = gather<Vector3f, 0, true>(ptr, idx10, active),
                     v01 = gather<Vector3f, 0, true>(ptr, idx01, active),
                     v11 = gather<Vector3f, 0, true>(ptr, idx11, active);
            Vector3f tmp = v01 + v10 - v11;

            du = (v10 + v00*(w.y() - 1.f) - tmp*w.y()) * Float(res);
            dv = (v01 + v00*(w.x() - 1.f) - tmp*w.x()) * Float(res);
        }
        return std::make_pair(du, dv);
#endif
    }

    /**
     * \brief Evaluates the covariance matrix of the LEAN representation
     * of the underlying normal map at a given level.
     */
    MTS_INLINE Matrix2f
    eval_lean_sigma(const Point2f &uv, size_t mip_level, Mask active=true) const {
        using Vector5f = Vector<Float, 5>;

        size_t res = resolution(mip_level);
        Float duv = rcp(Float(res));

        Point2f pos = (uv - 0.5f*duv)*res;
        pos = clamp(pos, Point2f(0), Point2f(res-1));
        Point2u p_min = Point2u(floor(pos)),
                p_max = p_min + 1;
        p_min = clamp(p_min, 0, res-1);
        p_max = clamp(p_max, 0, res-1);

        UInt32 idx00 = p_min.y()*res + p_min.x(),
               idx10 = p_min.y()*res + p_max.x(),
               idx01 = p_max.y()*res + p_min.x(),
               idx11 = p_max.y()*res + p_max.x();

        Point2f w1 = pos - p_min,
                w0 = 1.f - w1;

        const ScalarFloat *ptr = (const ScalarFloat *) m_lean_mipmap[mip_level]->data();
        Vector5f v00 = gather<Vector5f, 0, true>(ptr, idx00, active),
                 v10 = gather<Vector5f, 0, true>(ptr, idx10, active),
                 v01 = gather<Vector5f, 0, true>(ptr, idx01, active),
                 v11 = gather<Vector5f, 0, true>(ptr, idx11, active);
        Vector5f v0 = fmadd(w0.x(), v00, w1.x() * v10),
                 v1 = fmadd(w0.x(), v01, w1.x() * v11);
        Vector5f v = fmadd(w0.y(), v0, w1.y() * v1);

        Float a = max(0.f, v[2] - v[0]*v[0]),
              b = max(0.f, v[3] - v[1]*v[1]),
              c = max(0.f, v[4] - v[0]*v[1]);
        return Matrix2f(a, c, c, b);
    }

    MTS_INLINE Vector3f
    eval_smoothed_normal(const Point2f &uv, Float smoothing, bool use_slopes, Mask active=true) const {
        int L = levels();
        Float level = smoothing*(L - 1);
        size_t level0 = size_t(floor(level)),
               level1 = level0 + 1;
        level0 = clamp(level0, 0, L - 1);
        level1 = clamp(level1, 0, L - 1);

        Float w1 = level - level0,
              w0 = 1.f - w1;
        Vector3f n0 = eval_normal(uv, level0, use_slopes, active),
                 n1 = eval_normal(uv, level1, use_slopes, active);

        return fmadd(w0, n0, w1*n1);
    }

    MTS_INLINE std::pair<Vector3f, Vector3f>
    eval_smoothed_normal_derivatives(const Point2f &uv, Float smoothing, bool use_slopes, Mask active=true) const {
        int L = levels();
        Float level = smoothing*(L - 1);
        size_t level0 = size_t(floor(level)),
               level1 = level0 + 1;
        level0 = clamp(level0, 0, L - 1);
        level1 = clamp(level1, 0, L - 1);

        Float w1 = level - level0,
              w0 = 1.f - w1;

        auto [du0, dv0] = eval_normal_derivatives(uv, level0, use_slopes, active);
        auto [du1, dv1] = eval_normal_derivatives(uv, level1, use_slopes, active);

        Vector3f du = fmadd(w0, du0, w1*du1),
                 dv = fmadd(w0, dv0, w1*dv1);
        return std::make_pair(du, dv);
    }

    MTS_DECLARE_CLASS()
protected:
    std::string m_filename;

    std::vector<size_t> m_sizes;
    std::vector<ref<Bitmap>> m_normals_mipmap;      // Stores classical normal map information
    std::vector<ref<Bitmap>> m_lean_mipmap;         // Stores LEAN coefficients in 5 channels E[x], E[y], E[x^2], E[y^2], E[xy]
};

NAMESPACE_END(mitsuba)

#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>

NAMESPACE_BEGIN(mitsuba)

struct SMSConfig {
    bool biased    = false;                 // Switch from unbiased to biased SMS?
    bool twostage  = false;                 // Use two-stage solver for normal maps?
    bool halfvector_constraints = false;    // Switch back to original half-vector based constraints?
    bool mnee_init = false;                 // Use deterministic MNEE initialization instead?

    float step_scale = 1.f;                 // Scale step sizes inside Newton solver (mostly for visualizations)
    size_t max_iterations = 20;             // Maxiumum number of allowed iterations of the Newton solver
    float solver_threshold     = 1e-5f;     // Newton solver stopping criterion
    float uniqueness_threshold = 1e-4f;     // Threshold to distinguish unique solution paths (for pdf estimation)
    int max_trials = -1;                    // Trial set size M (for biased SMS), or upper limit of Bernoulli trials (for unbiased SMS)

    SMSConfig() {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SMSConfig[" << std::endl
            << "  biased = " << biased << "," << std::endl
            << "  twostage = " << twostage << "," << std::endl
            << "  halfvector_constraints = " << halfvector_constraints << "," << std::endl
            << "  mnee_init = " << mnee_init << "," << std::endl
            << "  step_scale = " << step_scale << "," << std::endl
            << "  max_iterations = " << max_iterations << "," << std::endl
            << "  solver_threshold = " << solver_threshold << "," << std::endl
            << "  uniqueness_threshold = " << uniqueness_threshold << "," << std::endl
            << "  max_trials       = " << max_trials << "," << std::endl
            << "]";
        return oss.str();
    }
};

template <typename Float_, typename Spectrum_>
struct ManifoldVertex {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_RENDER_BASIC_TYPES()
    using ShapePtr             = typename RenderAliases::ShapePtr;
    using SurfaceInteraction3f = typename RenderAliases::SurfaceInteraction3f;

    // Position and partials
    Point3f p;
    Vector3f dp_du, dp_dv;

    // Normal and partials
    Normal3f n, gn;
    Vector3f dn_du, dn_dv;

    // Tangents and partials
    Vector3f s, t;
    Vector3f ds_du, ds_dv;
    Vector3f dt_du, dt_dv;

    // Further information
    Float eta;
    Vector2f uv;
    ShapePtr shape;
    Mask fixed_direction;

    // Used in multi-bounce version
    Normal3f n_offset;
    Vector2f C;
    Matrix2f dC_dx_prev, dC_dx_cur, dC_dx_next;
    Matrix2f tmp, inv_lambda;
    Vector2f dx;

    ManifoldVertex(const Point3f &p = Point3f(0.f))
        : p(p), dp_du(0.f), dp_dv(0.f),
          n(0.f), gn(0.f), dn_du(0.f), dn_dv(0.f),
          s(0.f), t(0.f),
          ds_du(0.f), ds_dv(0.f), dt_du(0.f), dt_dv(0.f),
          eta(1.f), uv(0.f), shape(nullptr), fixed_direction(false) {}

    ManifoldVertex(const SurfaceInteraction3f &si, Float smoothing=0.f)
        : p(si.p), dp_du(si.dp_du), dp_dv(si.dp_dv),
          gn(si.n), uv(si.uv), shape(si.shape), fixed_direction(false) {

        // Encode conductors with eta=1.0, and dielectrics with their relative IOR
        Complex<Spectrum> ior = si.bsdf()->ior(si);
        eta = select(all(eq(0.f, imag(ior))), hmean(real(ior)), 1.f);     // Assumption here is that real (dielectric) IOR is not spectrally varying.

        // Compute frame and its derivative
        Frame3f frame = si.bsdf()->frame(si, smoothing);
        n = frame.n;
        s = frame.s;
        t = frame.t;

        auto [dframe_du, dframe_dv] = si.bsdf()->frame_derivative(si, smoothing);
        dn_du = dframe_du.n;
        dn_dv = dframe_dv.n;
        ds_du = dframe_du.s;
        ds_dv = dframe_dv.s;
        dt_du = dframe_du.t;
        dt_dv = dframe_dv.t;

        // In rare cases, e.g. 'twosided' materials, the geometric normal needs to be flipped
        masked(gn, dot(n, gn) < 0.f) *= -1.f;
    }

    void make_orthonormal() {
        // Turn into orthonormal parameterization at 'p'
        Float inv_norm = rcp(norm(dp_du));
        dp_du *= inv_norm;
        dn_du *= inv_norm;
        Float dp = dot(dp_du, dp_dv);
        Vector3f dp_dv_tmp = dp_dv - dp*dp_du;
        Vector3f dn_dv_tmp = dn_dv - dp*dn_du;
        inv_norm = rcp(norm(dp_dv_tmp));
        dp_dv = dp_dv_tmp * inv_norm;
        dn_dv = dn_dv_tmp * inv_norm;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "ManifoldVertex[" << std::endl
            << "  p = " << p << "," << std::endl
            << "  n = " << n << "," << std::endl
            << "  gn = " << gn << "," << std::endl
            << "  dp_du = " << dp_du << "," << std::endl
            << "  dp_dv = " << dp_dv << "," << std::endl
            << "  dn_du = " << dn_du << "," << std::endl
            << "  dn_dv = " << dn_dv << "," << std::endl
            << "  eta = " << eta << "," << std::endl
            << "  uv = " << uv << std::endl
            << "]";
        return oss.str();
    }
};

template <typename Float_, typename Spectrum_>
struct EmitterInteraction {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_RENDER_BASIC_TYPES()
    using EmitterPtr = typename RenderAliases::EmitterPtr;

    Point3f p;      // Emitter position (for area / point)
    Normal3f n;
    Vector3f d;     // Emitter direction (for infinite / directional )

    Spectrum weight;    // Samping weight (already divided by positional sampling pdf)
    Float pdf;          // Sampling pdf

    EmitterPtr emitter = nullptr;

    bool is_point() const {
        return has_flag(emitter->flags(), EmitterFlags::DeltaPosition);
    }

    bool is_directional() const {
        return has_flag(emitter->flags(), EmitterFlags::DeltaDirection) ||
               has_flag(emitter->flags(), EmitterFlags::Infinite);
    }

    bool is_area() const {
        return has_flag(emitter->flags(), EmitterFlags::Surface);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "EmitterInteraction[" << std::endl
            << "  p = " << p << "," << std::endl
            << "  n = " << n << "," << std::endl
            << "  d = " << d << "," << std::endl
            << "  weight = " << weight << "," << std::endl
            << "  pdf    = " << pdf << "," << std::endl
            << "]";
        return oss.str();
    }
};

template <typename Float_, typename Spectrum_>
struct SpecularManifold {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_TYPES(Sampler, Scene)
    using EmitterPtr         = typename RenderAliases::EmitterPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using ManifoldVertex     = ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;

    /// Sample emitter interaction for specular manifold sampling
    static EmitterInteraction
    sample_emitter_interaction(const Scene *scene,
                               const SurfaceInteraction3f &si,
                               ref<Sampler> sampler,
                               bool multi_scatter=false) {
        EmitterInteraction ei;
        Spectrum spec = 0.f;
        const auto emitters = multi_scatter ? scene->caustic_emitters_multi_scatter()
                                            : scene->caustic_emitters_single_scatter();

        if (unlikely(emitters.empty())) {
            Log(Warn, "Specular manifold sampling: no emitter is marked!");
            return ei;
        }

        // Uniformly sample an emitter, same was as in Scene::sample_emitter_direction
        Float emitter_sample = sampler->next_1d();
        Float emitter_pdf = 1.f / emitters.size();
        UInt32 index = min(UInt32(emitter_sample * (ScalarFloat) emitters.size()), (uint32_t) emitters.size()-1);
        EmitterPtr emitter = gather<EmitterPtr>(emitters.data(), index);
        ei.emitter = emitter;

        if (ei.is_area()) {
            ShapePtr shape = emitter->shape();
            PositionSample3f ps = shape->sample_position(si.time, sampler->next_2d());
            if (ps.pdf > 0) {
                SurfaceInteraction3f si_emitter;
                si_emitter.p = ps.p;
                si_emitter.wi = Vector3f(0.f, 0.f, 1.f);
                si_emitter.wavelengths = si.wavelengths;
                si_emitter.time = si.time;

                spec = emitter->eval(si_emitter) / ps.pdf;

                ei.p = ps.p;
                ei.n = ps.n;
                ei.d = normalize(ps.p - si.p);
                ei.pdf = ps.pdf;
            }
        } else if (ei.is_directional()) {
            auto [ds, spec_] = emitter->sample_direction(si, sampler->next_2d());
            ei.p = ds.p;
            ei.d = ds.d;
            ei.n = ds.d;
            ei.pdf = ds.pdf;
            spec = spec_;
        } else if (ei.is_point()) {
            auto [ds, spec_] = emitter->sample_direction(si, sampler->next_2d());
            ei.p = ds.p;
            ei.d = ds.d;
            ei.n = ei.d;
            ei.pdf = ds.pdf;
            // Remove solid angle conversion factor. This will be accounted for later in the geometric term computation.
            spec = spec_ * ds.dist*ds.dist;
        }

        ei.pdf *= emitter_pdf;
        ei.weight = spec * rcp(emitter_pdf);

        return ei;
    }

    /// Prepare emitter interaction for generalized geometry term computation
    static std::pair<Mask, ManifoldVertex>
    emitter_interaction_to_vertex(const Scene *scene,
                                  const EmitterInteraction &ei,
                                  const Point3f &p,
                                  Float time,
                                  const Wavelength &wavelengths) {
        if (ei.is_area()) {
            // Area emitters
            Vector3f d_tmp = normalize(ei.p - p);
            Ray3f ray_tmp(p + math::ShadowEpsilon<Float>*d_tmp, d_tmp, time, wavelengths);
            SurfaceInteraction3f si_y = scene->ray_intersect(ray_tmp);
            if (!si_y.is_valid()) {
                return std::make_pair(false, ManifoldVertex(Point3f(0.f)));
            }
            ManifoldVertex vy(si_y);
            vy.make_orthonormal();
            return std::make_pair(true, vy);
        } else if (ei.is_directional()) {
            // Directional & infinite emitters
            ManifoldVertex vy(ei.p);
            Vector3f d = normalize(ei.p - p);
            vy.p = p + d; // Place fake vertex at distance 1
            vy.n = -d;
            auto [s, t] = coordinate_system(vy.n);
            vy.dp_du = s;
            vy.dp_dv = t;
            vy.fixed_direction = true;
            return std::make_pair(true, vy);
        } else if (ei.is_point()) {
            // Point emitters
            ManifoldVertex vy(ei.p);
            Vector3f d = normalize(p - ei.p);
            vy.n = vy.gn = d;
            auto [s, t] = coordinate_system(d);
            vy.dp_du = s;
            vy.dp_dv = t;
            return std::make_pair(true, vy);
        }
        return std::make_pair(false, ManifoldVertex(Point3f(0.f)));
    }

    /// Convert SurfaceInteraction into a EmitterInteraction struct
    static EmitterInteraction
    emitter_interaction(const Scene *scene, const SurfaceInteraction3f &si, const SurfaceInteraction3f &si_emitter) {
        EmitterInteraction ei;

        EmitterPtr emitter = si_emitter.emitter(scene);
        ei.emitter = emitter;

        // Is either area light or infinite light, as it needs to be hit explicitly in the scene.
        ShapePtr shape = emitter->shape();
        if (shape) {
            ei.p = si_emitter.p;
            ei.n = si_emitter.n;
            ei.d = normalize(si_emitter.p - si.p);
        } else if (emitter->is_environment()) {
            ei.d = -si_emitter.wi;
            ei.p = si.p + 1000.f*ei.d;
            ei.n = si_emitter.wi;
        }
        return ei;
    }

    /// Sample the bivariate normal distribution for given mean vector and covariance matrix
    static MTS_INLINE
    Point2f sample_gaussian(const Point2f &mu, const Matrix2f &sigma, const Point2f &sample) {
        // Based on https://math.stackexchange.com/questions/268298/sampling-from-a-2d-normal-with-a-given-covariance-matrix
        Point2f p = warp::square_to_std_normal(sample);
        Float sigma_x = sqrt(sigma(0, 0)),
              sigma_y = sqrt(sigma(1, 1));

        Float rho = sigma(1, 0) / (sigma_x * sigma_y);
        Matrix2f P(0.5f, 0.5f,
                   0.5f, 0.5f);
        Matrix2f Q(0.5f, -0.5f,
                   -0.5f, 0.5f);
        Matrix2f A = sqrt(1.f + rho)*P + sqrt(1.f - rho)*Q;
        p = A*p;

        p[0] *= sigma_x;
        p[1] *= sigma_y;
        return p + mu;
    }

    static MTS_INLINE
    std::pair<Mask, Vector3f> reflect(const Vector3f &w, const Normal3f &n) {
        return std::make_pair(true, 2.f*dot(w, n)*n - w);
    }

    static MTS_INLINE
    std::pair<Vector3f, Vector3f> d_reflect(const Vector3f &w, const Vector3f &dw_du, const Vector3f &dw_dv,
                                            const Normal3f &n, const Vector3f &dn_du, const Vector3f &dn_dv) {
        Float dot_w_n    = dot(w, n),
              dot_dwdu_n = dot(dw_du, n),
              dot_dwdv_n = dot(dw_dv, n),
              dot_w_dndu = dot(w, dn_du),
              dot_w_dndv = dot(w, dn_dv);
        Vector3f dwr_du = 2.f*((dot_dwdu_n + dot_w_dndu)*n + dot_w_n*dn_du) - dw_du,
                 dwr_dv = 2.f*((dot_dwdv_n + dot_w_dndv)*n + dot_w_n*dn_dv) - dw_dv;
        return std::make_pair(dwr_du, dwr_dv);
    }

    static MTS_INLINE
    std::pair<Mask, Vector3f> refract(const Vector3f &w, const Normal3f &n_, Float eta_) {
        Normal3f n = n_;
        Float eta = rcp(eta_);
        if (dot(w, n) < 0) {
            // Coming from the "inside"
            eta = rcp(eta);
            n *= -1.f;
        }
        Float dot_w_n = dot(w, n);
        Float root_term = 1.f - eta*eta * (1.f - dot_w_n*dot_w_n);
        if (root_term < 0.f) {
            return std::make_pair(false, Vector3f(0.f));
        }
        Vector3f wt = -eta*(w - dot_w_n*n) - n*sqrt(root_term);
        return std::make_pair(true, wt);
    }

    static MTS_INLINE
    std::pair<Vector3f, Vector3f> d_refract(const Vector3f &w, const Vector3f &dw_du, const Vector3f &dw_dv,
                                            const Normal3f &n_, const Vector3f &dn_du_, const Vector3f &dn_dv_,
                                            Float eta_) {
        Normal3f n = n_;
        Vector3f dn_du = dn_du_,
                 dn_dv = dn_dv_;
        Float eta = rcp(eta_);
        if (dot(w, n) < 0) {
            // Coming from the "inside"
            eta = rcp(eta);
            n *= -1.f;
            dn_du *= -1.f;
            dn_dv *= -1.f;
        }
        Float dot_w_n    = dot(w, n),
              dot_dwdu_n = dot(dw_du, n),
              dot_dwdv_n = dot(dw_dv, n),
              dot_w_dndu = dot(w, dn_du),
              dot_w_dndv = dot(w, dn_dv);
        Float root = sqrt(1.f - eta*eta * (1.f - dot_w_n*dot_w_n));

        Vector3f a_u = -eta*(dw_du - ((dot_dwdu_n + dot_w_dndu)*n + dot_w_n*dn_du)),
                 b1_u = dn_du * root,
                 b2_u = n * rcp(2.f*root) * (-eta*eta*(-2.f*dot_w_n*(dot_dwdu_n + dot_w_dndu))),
                 b_u = -(b1_u + b2_u),
                 a_v = -eta*(dw_dv - ((dot_dwdv_n + dot_w_dndv)*n + dot_w_n*dn_dv)),
                 b1_v = dn_dv * root,
                 b2_v = n * rcp(2.f*root) * (-eta*eta*(-2.f*dot_w_n*(dot_dwdv_n + dot_w_dndv))),
                 b_v = -(b1_v + b2_v);

        Vector3f dwt_du = a_u + b_u,
                 dwt_dv = a_v + b_v;
        return std::make_pair(dwt_du, dwt_dv);
    }

    static MTS_INLINE
    std::pair<Float, Float> sphcoords(const Vector3f &w) {
        Float theta = safe_acos(w[2]);
        Float phi   = atan2(w[1], w[0]);
        if (phi < 0.f) {
            phi += 2.f*math::Pi<Float>;
        }
        return std::make_pair(theta, phi);
    }

    static MTS_INLINE
    std::tuple<Float, Float, Float, Float> d_sphcoords(const Vector3f &w,
                                                       const Vector3f &dw_du, const Vector3f &dw_dv) {
        Float d_acos = -rcp(safe_sqrt(1.f - w[2]*w[2]));
        Vector2f d_theta = d_acos * Vector2f(dw_du[2], dw_dv[2]);

        Float yx = w[1] / w[0];
        Float d_atan = rcp(1 + yx*yx);
        Vector2f d_phi = d_atan * Vector2f(w[0]*dw_du[1] - w[1]*dw_du[0],
                                           w[0]*dw_dv[1] - w[1]*dw_dv[0]) * rcp(w[0]*w[0]);

        return std::make_tuple(d_theta[0], d_phi[0], d_theta[1], d_phi[1]);
    }
};

NAMESPACE_END(mitsuba)

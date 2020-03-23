#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/core/math.h>
#include <enoki/special.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * Based on the LEAN/LEADR Mapping reference implementation by Jonathan Dupuy
 * https://github.com/jdupuy/dj_brdf
 */

template <typename Float_, typename Spectrum_>
struct LEANParameters {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_RENDER_BASIC_TYPES()

    Normal3f n;                             // Mean normal
    Float    ax, ay;                        // Slope scale
    Float    rho, sqrt_one_minus_rho_sqr;   // Correlation
    Float    tx_n, ty_n;                    // Slope offset
};

template <typename Float_, typename Spectrum_>
struct LEAN {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_RENDER_BASIC_TYPES()
    using LEANParameters = LEANParameters<Float, Spectrum>;

    static LEANParameters p_from_lean(const Vector2f &mu, const Matrix2f &sigma) {
        LEANParameters p;
        p.tx_n = mu[0]; p.ty_n = mu[1];
        p.ax = std::sqrt(2*sigma(0, 0)); p.ay = std::sqrt(2*sigma(1, 1));
        p.rho = 2.f*sigma(1, 0) / (p.ax * p.ay);
        p.sqrt_one_minus_rho_sqr = std::sqrt(1.f - p.rho*p.rho);
        p.n = normalize(Normal3f(-p.tx_n, -p.ty_n, 1.f));
        return p;
    }

    static LEANParameters p_from_lean_and_base(const Vector2f &mu, const Matrix2f &sigma,
                                               Float alpha_u, Float alpha_v) {
        // Convert mu, sigma to linear representation
        Float lean_e1 = mu[0]; Float lean_e2 = mu[1];
        Float lean_ax = std::sqrt(2*sigma(0, 0));
        Float lean_ay = std::sqrt(2*sigma(1, 1));
        Float rho = 2.f*sigma(1, 0) / (lean_ax * lean_ay);
        Float lean_e3 = 0.5f*lean_ax*lean_ax + lean_e1*lean_e1;
        Float lean_e4 = 0.5f*lean_ay*lean_ay + lean_e2*lean_e2;
        Float lean_e5 = 0.5f*rho*lean_ax*lean_ay + lean_e1*lean_e2;

        // Convert base roughness to linear representation
        Float base_e3 = 0.5f*alpha_u*alpha_u;
        Float base_e4 = 0.5f*alpha_v*alpha_v;

        Float e1 = lean_e1; Float e2 = lean_e2;
        Float e3 = lean_e3 + base_e3;   // Assume base_e1 = 0
        Float e4 = lean_e4 + base_e4;   // Assume base_e2 = 0
        Float e5 = lean_e5;

        LEANParameters p;
        p.tx_n = e1; p.ty_n = e2;
        Float tmp3 = std::max(Float(0), e3 - e1*e1);
        Float tmp4 = std::max(Float(0), e4 - e2*e2);
        p.ax = std::sqrt(2.f*tmp3);
        p.ay = std::sqrt(2.f*tmp4);
        p.rho = 2.f*(e5 - e1*e2) / (p.ax*p.ay);
        p.sqrt_one_minus_rho_sqr = std::sqrt(1.f - p.rho*p.rho);
        p.n = normalize(Normal3f(-p.tx_n, -p.ty_n, 1.f));
        return p;
    }

    static Float gaf(const Vector3f &h, const Vector3f &wi, const Vector3f &wo,
              const LEANParameters &params) {
        Float G1_i = g1(h, wi, params),
              G1_o = g1(h, wo, params);
        Float tmp = G1_i*G1_o;
        if (tmp > 0) {
            return tmp;
            return std::max(Float(0), tmp / (G1_o + G1_i - tmp));
        }
        return 0.f;
    }

    static Float ndf(const Vector3f &h, const LEANParameters &params) {
        if (h.z() > math::Epsilon<Float>) {
            Float tmp1 = h.z()*h.z();
            Float tmp2 = tmp1*tmp1;
            Float slope_x = -h.x() / h.z(),
                  slope_y = -h.y() / h.z();
            return p22(slope_x, slope_y, params) / tmp2;
        }
        return 0.f;
    }

    static Float vndf(const Vector3f &h, const Vector3f &wi,
                      const LEANParameters &params) {
        Float dp = dot(wi, h);
        if (dp > 0.f) {
            Float D = ndf(h, params);
            return dp * D / sigma(wi, params);
        }
        return 0.f;
    }

    static Vector3f sample_vndf(const Vector3f &wi, const Point2f &sample,
                                const LEANParameters &params) {
        Float a = wi.x()*params.ax + wi.y()*params.ay*params.rho;
        Float b = wi.y()*params.ay*params.sqrt_one_minus_rho_sqr;
        Float c = wi.z() - wi.x()*params.tx_n - wi.y()*params.ty_n;
        Vector3f wi_std = normalize(Vector3f(a, b, c));

        if (wi_std.z() > 0.f) {
            auto [tx_m, ty_m] = sample_vp22_std(wi_std, sample);
            Float tx_h = params.ax*tx_m + params.tx_n;
            Float choleski = params.rho*tx_m + params.sqrt_one_minus_rho_sqr*ty_m;
            Float ty_h = params.ay*choleski + params.ty_n;

            return normalize(Vector3f(-tx_h, -ty_h, 1));
        }
        return Vector3f(0, 0, 1);
    }

    static Float p22(Float slope_x, Float slope_y,
                     const LEANParameters &params) {
        slope_x -= params.tx_n;
        slope_y -= params.ty_n;

        Float nrm = params.ax*params.ay*params.sqrt_one_minus_rho_sqr;
        Float x = slope_x / params.ax;
        Float tmp1 = params.ax*slope_y - params.rho*params.ay*slope_x;
        Float tmp2 = params.ax*params.ay*params.sqrt_one_minus_rho_sqr;
        Float y = tmp1 / tmp2;

        return p22_std(x, y) / nrm;
    }

    static Float p22_std(Float x, Float y) {
        Float r = x*x + y*y;
        return math::InvPi<Float> * std::exp(-r);
    }

    static Float g1(const Vector3f &/* h */, const Vector3f &k,
                    const LEANParameters &params) {
        if (dot(k, params.n) > 0) {
            return k.z() / sigma(k, params);
        }
        return 0.f;
    }

    static Float sigma(const Vector3f &k,
                       const LEANParameters &params) {
        Float a = k.x()*params.ax + k.y()*params.ay*params.rho;
        Float b = k.y()*params.ay*params.sqrt_one_minus_rho_sqr;
        Float c = k.z() - k.x()*params.tx_n - k.y()*params.ty_n;
        Float nrm = std::sqrt(a*a + b*b + c*c);
        Vector3f k_prime = Vector3f(a, b, c) / nrm;
        return nrm * sigma_std(k_prime.z());
    }

    static Float sigma_std(Float cos_theta) {
        if (cos_theta == 1.f) return 1.f;
        Float sin_theta = std::sqrt(1.f - cos_theta * cos_theta);
        Float nu = cos_theta / sin_theta;
        Float tmp = std::exp(-nu*nu) * math::InvSqrtPi<Float>;
        return 0.5f*(cos_theta * (1.f + std::erf(nu)) + sin_theta * tmp);
    }

    static std::pair<Float, Float> sample_vp22_std(const Vector3f &wi, const Vector2f &sample) {
        Float cos_theta = wi.z();
        Float sin_theta = cos_theta < 1.f ? std::sqrt(1.f - cos_theta*cos_theta) : 0.f;
        Float tx = qf2(sample.x(), cos_theta, sin_theta);
        Float ty = qf1(sample.y());

        if (sin_theta == 0.f) {
            return std::make_pair(tx, ty);
        }
        Float nrm = 1.f / std::sqrt(wi.x()*wi.x() + wi.y()*wi.y());
        Float cos_phi = wi.x()*nrm;
        Float sin_phi = wi.y()*nrm;

        return std::make_pair(cos_phi*tx - sin_phi*ty,
                              sin_phi*tx + cos_phi*ty);
    }

    static Float qf1(Float u) {
        return erfinv(2.f*u - 1.f);
    }

    static Float qf2(Float u, Float cos_theta, Float sin_theta) {
        /* The original inversion routine from the paper contained
           discontinuities, which causes issues for QMC integration
           and techniques like Kelemen-style MLT. The following code
           performs a numerical inversion with better behavior */
        Float cot_theta = cos_theta / sin_theta;
        Float tan_theta = sin_theta / cos_theta;

        /* Search interval -- everything is parameterized
           in the erf() domain */
        Float a = -1, c = std::erf(cot_theta);
        u = std::max(u, (Float)1e-6f);

        /* Start with a good initial guess */
        /* We can do better (inverse of an approximation computed in Mathematica) */
        Float fit = 1 + cos_theta
                    * (-0.876f + cos_theta * (0.4265f - 0.0594f * cos_theta));
        Float b = c - (1 + c) * std::pow(1 - u, fit);

        /* Normalization factor for the CDF */
        Float normalization = 1 / (1 + c + math::InvSqrtPi<Float> *
            tan_theta * std::exp(-cot_theta * cot_theta));

        int it = 0;
        while (++it < 10) {
            /* Bisection criterion -- the oddly-looking
               boolean expression are intentional to check
               for NaNs at little additional cost */
            if (!(b >= a && b <= c))
                b = 0.5f * (a + c);

            /* Evaluate the CDF and its derivative
               (i.e. the density function) */
            Float inv_erf = erfinv(b);
            Float value = normalization * (1 + b + math::InvSqrtPi<Float> *
                tan_theta * std::exp(-inv_erf * inv_erf)) - u;
            Float derivative = normalization * (1 - inv_erf * tan_theta);

            if (std::abs(value) < 1e-5f)
                break;

            /* Update bisection intervals */
            if (value > 0)
                c = b;
            else
                a = b;

            b -= value / derivative;
        }

        /* Now convert back into a slope value */
        return erfinv(std::max(-(Float)0.9999f, b));
    }
};

NAMESPACE_END(mitsuba)

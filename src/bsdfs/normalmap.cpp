#include <mitsuba/core/properties.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/normalmap.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/lean.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class NormalmapBSDF final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, component_count, m_components, m_flags)
    MTS_IMPORT_TYPES(Texture, BSDF)
    using Normalmap = Normalmap<Float, Spectrum>;
    using LEANParameters = LEANParameters<Float, Spectrum>;
    using LEAN = LEAN<Float, Spectrum>;

    NormalmapBSDF(const Properties &props) : Base(props) {
        for (auto &kv : props.objects()) {
            auto *bsdf = dynamic_cast<Base *>(kv.second.get());
            if (bsdf) {
                if (m_nested_bsdf)
                    Throw("Cannot specify more than one child BSDF");
                m_nested_bsdf = bsdf;
            }
        }
        if (!m_nested_bsdf)
           Throw("Child BSDF not specified");

        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_filename = file_path.filename().string();
        Log(Info, "Loading normalmap texture from \"%s\" ..", m_filename);
        m_normalmap = new Normalmap(file_path);
        m_total_levels = m_normalmap->levels();

        m_tiles = props.float_("tiles", 1.f);
        m_use_slopes = props.bool_("use_slopes", true);

        m_lean_fallback = props.bool_("lean_fallback", false);
        m_alpha_u = m_alpha_v = props.float_("lean_fallback_alpha", 0.0001f);
        std::string material = props.string("lean_fallback_material", "none");
        if (props.has_property("eta") || material == "none") {
            m_eta = props.texture<Texture>("eta", 0.f);
            m_k   = props.texture<Texture>("k",   1.f);
            if (material != "none")
                Throw("Should specify either (eta, k) or material, not both.");
        } else {
            std::tie(m_eta, m_k) = complex_ior_from_file<Spectrum, Texture>(props.string("lean_fallback_material", "none"));
        }

        parameters_changed({});
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        m_components.clear();
        for (size_t i = 0; i < m_nested_bsdf->component_count(); ++i)
            m_components.push_back(m_nested_bsdf->flags(i));

        m_flags = m_nested_bsdf->flags() | m_components.back();
        m_flags = m_flags | BSDFFlags::NeedsDifferentials;
        if (m_lean_fallback) {
            m_flags = m_flags | BSDFFlags::GlossyReflection;
        }
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        if (m_lean_fallback && !si.has_uv_partials()) {
            BSDFSample3f bs;
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection))) {
                return { bs, 0.f };
            }
            Float cos_theta_i = Frame3f::cos_theta(si.wi);
            active &= cos_theta_i > 0.f;

            auto [mu, sigma] = lean(si, active);
            LEANParameters params = LEAN::p_from_lean_and_base(mu, sigma, m_alpha_u, m_alpha_v);
            Normal3f m = LEAN::sample_vndf(si.wi, sample2, params);

            bs.wo = reflect(si.wi, m);
            bs.eta = 1.f;
            bs.sampled_component = 0;
            bs.sampled_type = +BSDFFlags::GlossyReflection;

            Float G = LEAN::gaf(m, bs.wo, si.wi, params);
            active &= G > 0.f && Frame3f::cos_theta(bs.wo) > 0.f;

            Complex<Spectrum> eta_c(m_eta->eval(si, active),
                                    m_k->eval(si, active));
            Spectrum F;
            if (all(eq(imag(eta_c), 0.f))) {
                auto [F_, cos_theta_t, eta_it, eta_ti] =
                    fresnel(Spectrum(dot(si.wi, m)), real(eta_c));
                F = F_;
            } else {
                F = fresnel_conductor(Spectrum(dot(si.wi, m)), eta_c);
            }

            Float G1 = LEAN::g1(m, si.wi, params);
            bs.pdf = LEAN::vndf(m, si.wi, params) / (4.f*dot(si.wi, m));
            Spectrum weight = F * G / G1;

            return { bs, weight & active };
        }

        // Sample nested BSDF with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(perturbed_si, 0.f, active);
        perturbed_si.wi = perturbed_si.to_local(si.to_world(si.wi));

        auto [bs, weight] = m_nested_bsdf->sample(ctx, perturbed_si,
                                                  sample1, sample2, active);
        active &= any(neq(weight, 0.f));
        if (none(active)) {
            return { bs, 0.f };
        }

        // Transform sampled wo back to original frame and check orientation
        Vector3f perturbed_wo = si.to_local(perturbed_si.to_world(bs.wo));
        active &= Frame3f::cos_theta(bs.wo)*Frame3f::cos_theta(perturbed_wo) > 0.f;
        bs.wo = perturbed_wo;

        return { bs, weight & active };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (m_lean_fallback && !si.has_uv_partials()) {
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection))) {
                return 0.f;
            }
            Float cos_theta_i = Frame3f::cos_theta(si.wi),
                  cos_theta_o = Frame3f::cos_theta(wo);
            active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

            auto [mu, sigma] = lean(si, active);
            LEANParameters params = LEAN::p_from_lean_and_base(mu, sigma, m_alpha_u, m_alpha_v);

            Vector3f H = normalize(wo + si.wi);

            Complex<Spectrum> eta_c(m_eta->eval(si, active),
                                    m_k->eval(si, active));
            Spectrum F;
            if (all(eq(imag(eta_c), 0.f))) {
                auto [F_, cos_theta_t, eta_it, eta_ti] =
                    fresnel(Spectrum(dot(si.wi, H)), real(eta_c));
                F = F_;
            } else {
                F = fresnel_conductor(Spectrum(dot(si.wi, H)), eta_c);
            }

            Float G = LEAN::gaf(H, si.wi, wo, params);
            Float D = LEAN::ndf(H, params);

            return F * D * G / (4.f * cos_theta_i);
        }

        // Evaluate nested BSDF with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(perturbed_si, 0.f, active);
        perturbed_si.wi       = perturbed_si.to_local(si.to_world(si.wi));
        Vector3f perturbed_wo = perturbed_si.to_local(si.to_world(wo));

        active &= Frame3f::cos_theta(wo)*Frame3f::cos_theta(perturbed_wo) > 0.f;

        return m_nested_bsdf->eval(ctx, perturbed_si,
                                   perturbed_wo, active);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!si.has_uv_partials() && m_lean_fallback) {
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection))) {
                return 0.f;
            }

            Float cos_theta_i = Frame3f::cos_theta(si.wi),
                  cos_theta_o = Frame3f::cos_theta(wo);
            active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

            auto [mu, sigma] = lean(si, active);
            LEANParameters params = LEAN::p_from_lean_and_base(mu, sigma, m_alpha_u, m_alpha_v);

            Vector3f H = normalize(wo + si.wi);

            Float G = LEAN::gaf(H, wo, si.wi, params);
            return select(G > 0.f, LEAN::vndf(H, si.wi, params) / (4.f*dot(si.wi, H)), 0.f);
        }

        // Evaluate nested BSDF with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(perturbed_si, 0.f, active);
        perturbed_si.wi       = perturbed_si.to_local(si.to_world(si.wi));
        Vector3f perturbed_wo = perturbed_si.to_local(si.to_world(wo));

        active &= Frame3f::cos_theta(wo)*Frame3f::cos_theta(perturbed_wo) > 0.f;

        return m_nested_bsdf->pdf(ctx, perturbed_si,
                                  perturbed_wo, active);
    }

    Frame3f frame(const SurfaceInteraction3f &si, Float smoothing,
                  Mask active) const override {
        Frame3f frame = m_nested_bsdf->frame(si, smoothing, active);

        Point2f uv = si.uv;
        uv *= m_tiles;
        uv -= floor(uv);

        Normal3f n;
        if (m_use_slopes) {
            n = Normal3f(m_normalmap->eval_smoothed_normal(uv, smoothing, true, active));
        } else {
            Vector3f rgb = m_normalmap->eval_smoothed_normal(uv, smoothing, false, active);
            n = 2.f*rgb - 1.f;
        }

        Frame3f result;
        result.n = normalize(frame.to_world(n));
        result.s = normalize(si.dp_du - result.n*dot(result.n, si.dp_du));
        result.t = cross(result.n, result.s);
        return result;
    }

    std::pair<Frame3f, Frame3f>
    frame_derivative(const SurfaceInteraction3f &si, Float smoothing,
                     Mask active) const override {
        Point2f uv = si.uv;
        uv *= m_tiles;
        uv -= floor(uv);

        Normal3f n;
        Vector3f dn_du, dn_dv;
        if (m_use_slopes) {
            Vector3f slope = m_normalmap->eval_smoothed_normal(uv, smoothing, true, active);
            Float inv_norm = rcp(norm(slope));
            n = slope * inv_norm;
            auto [dslope_du, dslope_dv] = m_normalmap->eval_smoothed_normal_derivatives(uv, smoothing, true, active);
            dn_du = inv_norm * (dslope_du - n*dot(n, dslope_du));
            dn_dv = inv_norm * (dslope_dv - n*dot(n, dslope_dv));
        } else {
            Vector3f rgb = m_normalmap->eval_smoothed_normal(uv, smoothing, false, active);
            auto [drgb_du, drgb_dv] = m_normalmap->eval_smoothed_normal_derivatives(uv, smoothing, false, active);
            n = 2.f*rgb - 1.f;
            dn_du = 2*drgb_du;
            dn_dv = 2*drgb_dv;
        }
        // Scale to account for tiling scale
        dn_du *= m_tiles;
        dn_dv *= m_tiles;

        Frame3f base = m_nested_bsdf->frame(si, smoothing, active);
        auto [dbase_du, dbase_dv] = m_nested_bsdf->frame_derivative(si, smoothing, active);

        Vector3f world_n = base.to_world(n);
        Float inv_length_n = rcp(norm(world_n));
        world_n *= inv_length_n;

        Frame3f dframe_du, dframe_dv;
        dframe_du.n = inv_length_n * (base.to_world(dn_du) + dbase_du.to_world(n));
        dframe_dv.n = inv_length_n * (base.to_world(dn_dv) + dbase_dv.to_world(n));
        dframe_du.n -= world_n*dot(dframe_du.n, world_n);
        dframe_dv.n -= world_n*dot(dframe_dv.n, world_n);

        Vector3f s = si.dp_du - world_n*dot(world_n, si.dp_du);
        Float inv_length_s = rcp(norm(s));
        s *= inv_length_s;

        dframe_du.s = inv_length_s * (-dframe_du.n*dot(world_n, si.dp_du) - world_n * dot(dframe_du.n, si.dp_du));
        dframe_dv.s = inv_length_s * (-dframe_dv.n*dot(world_n, si.dp_du) - world_n * dot(dframe_dv.n, si.dp_du));
        dframe_du.s -= s*dot(dframe_du.s, s);
        dframe_dv.s -= s*dot(dframe_dv.s, s);

        dframe_du.t = cross(dframe_du.n, s) + cross(world_n, dframe_du.s);
        dframe_dv.t = cross(dframe_dv.n, s) + cross(world_n, dframe_dv.s);

        return { dframe_du, dframe_dv };
    }

    std::pair<Point2f, Matrix2f> lean(const SurfaceInteraction3f &si, Mask active) const override {
        size_t level = mipmap_level(si);

        Point2f uv = si.uv;
        uv *= m_tiles;
        uv -= floor(uv);

        Vector3f n = m_normalmap->eval_normal(uv, level, m_use_slopes, active);
        Point2f mu(-n[0]/n[2], -n[1]/n[2]);
        Matrix2f sigma = m_normalmap->eval_lean_sigma(uv, level, true);
        return std::make_pair(mu, sigma);
    }

    Point2f slope(const Point2f &uv_, Mask active) const override {
        if (unlikely(!m_use_slopes)) {
            Log(Warn, "Normalmap: For glint rendering, only slope-based normal maps are supported.");
        }

        Point2f uv = uv_;
        uv *= m_tiles;
        uv -= floor(uv);

        Normal3f n = m_normalmap->eval_normal(uv, 0, true, active);
        return Point2f(-n[0], -n[1]);   // n[2] == 1
    }

    std::pair<Vector2f, Vector2f>
    slope_derivative(const Point2f &uv_, Mask active) const override {
        if (unlikely(!m_use_slopes)) {
            Log(Warn, "Normalmap: For glint rendering, only slope-based normal maps are supported.");
        }

        Point2f uv = uv_;
        uv *= m_tiles;
        uv -= floor(uv);

        auto [du_, dv_] = m_normalmap->eval_normal_derivatives(uv, 0, true, active);
        Vector2f du(du_[0], du_[1]);
        Vector2f dv(dv_[0], dv_[1]);
        // Scale to account for tiling scale
        du *= m_tiles;
        dv *= m_tiles;

        return { du, dv };
    }

    Float roughness() const override {
        return m_nested_bsdf->roughness();
    }

    Complex<Spectrum> ior(const SurfaceInteraction3f &si, Mask active) const override {
        return m_nested_bsdf->ior(si, active);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Normalmap[" << std::endl
            << "  filename = " << m_filename << "," << std::endl
            << "  nested_bsdf = " << string::indent(m_nested_bsdf->to_string()) << std::endl
            << "]";
        return oss.str();
    }

protected:
    size_t mipmap_level(const SurfaceInteraction3f &si) const {
        if (!si.has_uv_partials()) {
            return m_total_levels - 1;
        }

        Vector2f duv_dx = m_tiles*si.duv_dx,
                 duv_dy = m_tiles*si.duv_dy;

        Float width = 2*std::max(std::max(std::abs(duv_dx[0]), std::abs(duv_dx[1])),
                                 std::max(std::abs(duv_dy[0]), std::abs(duv_dy[1])));

        size_t level = size_t(std::ceil(m_total_levels - 1 + std::log2(std::max(width, Float(1e-8f)))));
        return std::min(level, m_total_levels - 1);
    }

    MTS_DECLARE_CLASS()
private:
    // Information about underlying normal map data
    std::string m_filename;
    ref<Normalmap> m_normalmap;
    size_t m_total_levels;

    // Child BSDF that is affected by the normal map
    ref<BSDF> m_nested_bsdf;

    // Information about how data should be accessed
    Float m_tiles;
    bool m_use_slopes;

    // Optionally, this normal map can "fall back" to using the highest LEAN
    // mapping level on secondary bounces (or in general when no ray differential
    // informations is available)
    bool m_lean_fallback;
    Float m_alpha_u, m_alpha_v;
    ref<Texture> m_eta;
    ref<Texture> m_k;
};

MTS_IMPLEMENT_CLASS_VARIANT(NormalmapBSDF, BSDF)
MTS_EXPORT_PLUGIN(NormalmapBSDF, "Normal mapped material")
NAMESPACE_END(mitsuba)

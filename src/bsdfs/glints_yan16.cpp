#include <mitsuba/core/string.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/flake.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/flake.h>
#include <enoki/stl.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * Code for "Position-Normal Distributions for Efficient Rendering of Specular Microstructure"
 * by Ling-Qi Yan, Miloš Hašan, Steve Marschner, and Ravi Ramamoorthi.
 * ACM Transactions on Graphics (Proceedings of SIGGRAPH 2016)
 *
 * Released on:
 * https://sites.cs.ucsb.edu/~lingqi/
 *
 * With minor adapations to connect it with the Mitsuba 2 infrastructure and
 * adjusted sample/eval/pdf methods for the case where ray differential information
 * si.duv_dx and si.duv_dy are not available (e.g. for indirect bounces).
 * In that case, the implementation falls back to a standard Microfacet model.
 */
template <typename Float, typename Spectrum>
class YanGlintyBSDF final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture, MicrofacetDistribution)

    YanGlintyBSDF(const Properties &props) : Base(props) {
        if (props.has_property("distribution")) {
            std::string distr = string::to_lower(props.string("distribution"));
            if (distr == "beckmann")
                m_type = MicrofacetType::Beckmann;
            else if (distr == "ggx")
                m_type = MicrofacetType::GGX;
            else
                Throw("Specified an invalid distribution \"%s\", must be "
                      "\"beckmann\" or \"ggx\"!", distr.c_str());
        } else {
            m_type = MicrofacetType::Beckmann;
        }

        m_sample_visible = props.bool_("sample_visible", true);

        if (props.has_property("alpha_u") || props.has_property("alpha_v")) {
            if (!props.has_property("alpha_u") || !props.has_property("alpha_v"))
                Throw("Microfacet model: both 'alpha_u' and 'alpha_v' must be specified.");
            if (props.has_property("alpha"))
                Throw("Microfacet model: please specify"
                      "either 'alpha' or 'alpha_u'/'alpha_v'.");
            m_alpha_u = props.float_("alpha_u");
            m_alpha_v = props.float_("alpha_v");
        } else {
            m_alpha_u = m_alpha_v = props.float_("alpha", 0.1f);
        }

        m_intrinsic = props.float_("intrinsic", 0.001f);
        m_tiles = props.float_("tiles", 1.f);
        m_scale = props.float_("scale", 1.f);

        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_filename = file_path.filename().string();
        Log(Info, "Loading normalmap texture from \"%s\" ..", m_filename);

        std::string path_str = file_path.string();

        Timer timer;
        Log(Info, "Initialize flakes hierarchy..");
        timer.reset();
        m_flakes_tree.initialize(path_str.c_str(), m_intrinsic);
        Log(Info, "done. (took %s)",
            util::time_string(timer.value(), true));

        parameters_changed({});
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide | BSDFFlags::NeedsDifferentials;
        if (m_alpha_u != m_alpha_v)
            m_flags = m_flags | BSDFFlags::Anisotropic;

        m_components.clear();
        m_components.push_back(m_flags);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /*sample1*/,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return { 0.f, 0.f };
        } else {
            BSDFSample3f bs;
            Float cos_theta_i = Frame3f::cos_theta(si.wi);
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || cos_theta_i < 0)) {
                return { bs, 0.f };
            }

            /* Construct a microfacet distribution matching the
               roughness values at the current surface position. */
            MicrofacetDistribution distr(m_type,
                                         m_alpha_u, m_alpha_v,
                                         m_sample_visible);

            Normal3f m;
            if (!si.has_uv_partials()) {
                Float unused;
                std::tie(m, unused) = distr.sample(si.wi, sample2);
            } else {
                m = sample_ndf(si, (Sampler<Float, Spectrum> *)ctx.sampler);
            }

            if (m[0] == 0.f && m[1] == 0.f && m[2] == 0.f) {
                return { bs, 0.f };
            }

            bs.wo = reflect(si.wi, m);
            bs.eta = 1.f;
            bs.sampled_component = 0;
            bs.sampled_type = +BSDFFlags::GlossyReflection;
            if (!si.has_uv_partials()) {
                bs.pdf = pdf(ctx, si, bs.wo, active);
            } else {
                bs.pdf = eval_ndf(m, si) / (4.f * fabsf(dot(m, bs.wo))) * Frame3f::cos_theta(m);
            }
            if (bs.pdf == 0.f) {
                return { bs, 0.f };
            }

            if (Frame3f::cos_theta(bs.wo) < 0) {
                return { bs, 0.f };
            }

            /* Evaluate Smith's shadow-masking function */
            Float G = distr.G(si.wi, bs.wo, m);

            /* Calculate the total amount of reflection */
            Float model = G / (4.0f * cos_theta_i);
            Spectrum weight = model * (4.0f * fabsf(dot(m, bs.wo))) / Frame3f::cos_theta(m);
            return { bs, weight };
        }
    }

    Spectrum eval(const BSDFContext &ctx,
                  const SurfaceInteraction3f &si,
                  const Vector3f &wo,
                  Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return 0.f;
        } else {
            Float cos_theta_i = Frame3f::cos_theta(si.wi),
                  cos_theta_o = Frame3f::cos_theta(wo);
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                         cos_theta_i < 0.f ||
                         cos_theta_o < 0.f)) {
                return 0;
            }

            /* Calculate the half-direction vector */
            Vector3f H = normalize(wo + si.wi);

            /* Construct a microfacet distribution matching the
               roughness values at the current surface position. */
            MicrofacetDistribution distr(m_type,
                                         m_alpha_u, m_alpha_v,
                                         m_sample_visible);

            Float D = 0.f;
            if (!si.has_uv_partials()) {
                D = distr.eval(H);
            } else {
                D = eval_ndf(H, si);
            }

            if (D == 0.f) {
                return 0.f;
            }

            /* Evaluate Smith's shadow-masking function */
            Float G = distr.G(si.wi, wo, H);

            /* Evaluate the full microfacet model */
            return D * G / (4.f * cos_theta_i);
        }
    }

    Float pdf(const BSDFContext &ctx,
              const SurfaceInteraction3f &si,
              const Vector3f &wo,
              Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return 0.f;
        } else {
            Float cos_theta_i = Frame3f::cos_theta(si.wi),
                  cos_theta_o = Frame3f::cos_theta(wo);
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
                         cos_theta_i < 0.f ||
                         cos_theta_o < 0.f)) {
                return 0;
            }

            /* Calculate the half-direction vector */
            Vector3f H = normalize(wo + si.wi);

            /* Construct a microfacet distribution matching the
               roughness values at the current surface position. */
            MicrofacetDistribution distr(m_type,
                                         m_alpha_u, m_alpha_v,
                                         m_sample_visible);

            Float D = 0.f;
            if (!si.has_uv_partials()) {
                D = distr.eval(H);
            } else {
                D = eval_ndf(H, si);
            }

            if (D == 0.f) {
                return 0.f;
            }

            return D / (4.f * fabsf(dot(H, wo))) * Frame3f::cos_theta(H);
        }
    }

    void constrain_prf(Float &sigma_p1, Float &sigma_p2) const {
        const int MAX_QUERY_RANGE = 128;
        if (sigma_p1 * FLAKE_PIXEL_SIGMAS * m_flakes_tree.m_resolutionU > MAX_QUERY_RANGE)
            sigma_p1 = MAX_QUERY_RANGE / FLAKE_PIXEL_SIGMAS / m_flakes_tree.m_resolutionU;
        if (sigma_p2 * FLAKE_PIXEL_SIGMAS * m_flakes_tree.m_resolutionV > MAX_QUERY_RANGE)
            sigma_p2 = MAX_QUERY_RANGE / FLAKE_PIXEL_SIGMAS / m_flakes_tree.m_resolutionV;
    }

    Float normrnd(Float mu, Float sigma, Sampler<Float, Spectrum> *sampler) const {
        Float U = clamp(sampler->next_1d(), 1e-20f, 1.0f - 1e-20f);
        Float V = clamp(sampler->next_1d(), 1e-20f, 1.0f - 1e-20f);
        Float X = sqrt(-2.0f * log(U)) * cos(2.0f * math::Pi<Float> * V);
        return X * sigma + mu;
    }

    Float eval_ndf(const Vector3f &h, const SurfaceInteraction3f &si) const {
        Float D = 0.0f;

        Float x1 = si.uv.y() * m_tiles;    // In Mitsuba, u and v are along column and row, respectively. So we swap them.
        Float x2 = si.uv.x() * m_tiles;

        Float du = (fabs(si.duv_dx[0]) + fabs(si.duv_dy[0])) * m_scale;
        Float dv = (fabs(si.duv_dx[1]) + fabs(si.duv_dy[1])) * m_scale;
        Float sigma_p1 = dv / 2.f * m_tiles;
        Float sigma_p2 = du / 2.f * m_tiles;

        constrain_prf(sigma_p1, sigma_p2);

        int x1a = (int) floor(x1 - FLAKE_PIXEL_SIGMAS * sigma_p1);
        int x1b = (int) floor(x1 + FLAKE_PIXEL_SIGMAS * sigma_p1);
        int x2a = (int) floor(x2 - FLAKE_PIXEL_SIGMAS * sigma_p2);
        int x2b = (int) floor(x2 + FLAKE_PIXEL_SIGMAS * sigma_p2);

        for (int i = x1a; i <= x1b; i++) {
            for (int j = x2a; j <= x2b; j++) {
                std::vector<Flake*> candidate_flakes;
                candidate_flakes.reserve(256);
                Float x1Prime = x1 - i;
                Float x2Prime = x2 - j;
                Vector4f aa((x1Prime - FLAKE_PIXEL_SIGMAS * sigma_p1) * m_flakes_tree.m_resolutionU, (x2Prime - FLAKE_PIXEL_SIGMAS * sigma_p2) * m_flakes_tree.m_resolutionV, h[0], h[1]);
                Vector4f bb((x1Prime + FLAKE_PIXEL_SIGMAS * sigma_p1) * m_flakes_tree.m_resolutionU, (x2Prime + FLAKE_PIXEL_SIGMAS * sigma_p2) * m_flakes_tree.m_resolutionV, h[0], h[1]);
                m_flakes_tree.queryFlakesEval(aa, bb, candidate_flakes);
                for (size_t k = 0; k < candidate_flakes.size(); k++) {
                    D += candidate_flakes[k]->contributionToNdf(Vector2f(x1Prime * m_flakes_tree.m_resolutionU, x2Prime * m_flakes_tree.m_resolutionV), Vector2f(sigma_p1 * m_flakes_tree.m_resolutionU, sigma_p2 * m_flakes_tree.m_resolutionV), Vector2f(h[0], h[1]), m_intrinsic);
                }
            }
        }

        return D;
    }

    Normal3f sample_ndf(const SurfaceInteraction3f &si, Sampler<Float, Spectrum> *sampler) const {
        Float x1 = si.uv.y() * m_tiles;    // In Mitsuba, u and v are along column and row, respectively. So we swap them.
        Float x2 = si.uv.x() * m_tiles;

        Float du = (fabs(si.duv_dx[0]) + fabs(si.duv_dy[0])) * m_scale;
        Float dv = (fabs(si.duv_dx[1]) + fabs(si.duv_dy[1])) * m_scale;
        Float sigma_p1 = dv / 2.f * m_tiles;
        Float sigma_p2 = du / 2.f * m_tiles;

        constrain_prf(sigma_p1, sigma_p2);

        Float x1Sample = normrnd(x1, sigma_p1, sampler);
        Float x2Sample = normrnd(x2, sigma_p2, sampler);
        x1Sample -= floor(x1Sample);
        x2Sample -= floor(x2Sample);
        Float u1Sample = x1Sample * m_flakes_tree.m_resolutionU;
        Float u2Sample = x2Sample * m_flakes_tree.m_resolutionV;

        Vector4f aa(u1Sample, u2Sample, -1.0f, -1.0f);
        Vector4f bb(u1Sample, u2Sample, 1.0f, 1.0f);
        std::vector<Flake*> candidate_flakes;
        candidate_flakes.reserve(256);
        m_flakes_tree.queryFlakesSample(aa, bb, candidate_flakes);
        std::vector<Float> candidateWeights;
        Float sumWeight = 0.0f;
        for (size_t i = 0; i < candidate_flakes.size(); i++) {
            Float weight = g(candidate_flakes[i]->u0[0], u1Sample, candidate_flakes[i]->shape[0]) *
                           g(candidate_flakes[i]->u0[1], u2Sample, candidate_flakes[i]->shape[1]) *
                           candidate_flakes[i]->area;

            candidateWeights.push_back(weight);
            sumWeight += weight;
        }
        if (sumWeight == 0.0f) {
            // TODO: Think more about the ``blank area''.
            return Normal3f(0.0f, 0.0f, 0.0f);
        }

        // TODO: This can be done more efficiently.
        Float randNum = sampler->next_1d();
        Flake *selectedFlake;
        for (size_t i = 0; i < candidate_flakes.size(); i++) {
            randNum -= candidateWeights[i] / sumWeight;
            if (randNum <= 0.0f) {
                selectedFlake = candidate_flakes[i];
                break;
            }
        }
        if (randNum > 0.0f)
            selectedFlake = candidate_flakes.back();

        Vector2f nSample = selectedFlake->getNormal(Vector2f(u1Sample, u2Sample));
        Float n1Sample = normrnd(nSample[0], m_intrinsic, sampler);
        Float n2Sample = normrnd(nSample[1], m_intrinsic, sampler);
        Float n3SampleSqr = 1.0f - n1Sample * n1Sample - n2Sample * n2Sample;
        if (n3SampleSqr < 0.0f) {
            return Normal3f(0.0f, 0.0f, 0.0f);
        }
        Float n3Sample = sqrtf(n3SampleSqr);

        return Normal3f(n1Sample, n2Sample, n3Sample);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "YanGlintyBSDF[]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    /// Specifies the type of microfacet distribution
    MicrofacetType m_type;
    /// Anisotropic roughness values
    Float m_alpha_u, m_alpha_v;
    /// Importance sample the distribution of visible normals?
    bool m_sample_visible;

    /// Normal/flake map filename
    std::string m_filename;
    FlakesTree m_flakes_tree;

    Float m_intrinsic;
    Float m_tiles;
    Float m_scale;
};

MTS_IMPLEMENT_CLASS_VARIANT(YanGlintyBSDF, BSDF)
MTS_EXPORT_PLUGIN(YanGlintyBSDF, "Yan et al. 2016 glinty BSDF");

NAMESPACE_END(mitsuba)
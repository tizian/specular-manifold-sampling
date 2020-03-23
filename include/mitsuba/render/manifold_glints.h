#pragma once

#include <mitsuba/render/manifold.h>

NAMESPACE_BEGIN(mitsuba)

/// Datastructure handling specular manifold sampling in the multi-bounce case.
template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER SpecularManifoldGlints : public Object {
public:
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape)
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using ManifoldVertex     = ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;
    using SpecularManifold   = SpecularManifold<Float, Spectrum>;

    /// Initialize data structure
    SpecularManifoldGlints(const Scene *scene, const SMSConfig &config);

    virtual ~SpecularManifoldGlints();

    // ========================================================================
    //           Main functionality, to be called from integrators
    // ========================================================================



    // ========================================================================
    //           Helper functions for internal use, or debugging
    // ========================================================================


    static void print_statistics();

protected:
    static std::atomic<int> stats_mfw_failed;
    static std::atomic<int> stats_mfw_succeeded;
    static std::atomic<int> stats_bernoulli_trial_calls;
    static std::atomic<int> stats_bernoulli_trial_iterations;
    static std::atomic<int> stats_bernoulli_trial_iterations_max;

    MTS_DECLARE_CLASS()
protected:
    const Scene *m_scene = nullptr;
    SMSConfig m_config;
};

NAMESPACE_END(mitsuba)

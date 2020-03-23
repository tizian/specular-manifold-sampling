#include <mitsuba/render/manifold_ms.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sampler.h>
#include <iomanip>

NAMESPACE_BEGIN(mitsuba)

template<typename T>
inline void update_max(std::atomic<T> & atom, const T val) {
  for(T atom_val=atom; atom_val < val && !atom.compare_exchange_weak(atom_val, val, std::memory_order_relaxed););
}

MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_mfw_failed(0);
MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_mfw_succeeded(0);
MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_bernoulli_trial_calls(0);
MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_bernoulli_trial_iterations(0);
MTS_VARIANT std::atomic<int> SpecularManifoldMultiScatter<Float, Spectrum>::stats_bernoulli_trial_iterations_max(0);

MTS_VARIANT SpecularManifoldMultiScatter<Float, Spectrum>::SpecularManifoldMultiScatter(
    const Scene *scene, const SMSConfig &config) {
    m_scene = scene;
    m_config = config;
}

MTS_VARIANT SpecularManifoldMultiScatter<Float, Spectrum>::~SpecularManifoldMultiScatter() {}



MTS_VARIANT void SpecularManifoldMultiScatter<Float, Spectrum>::print_statistics() {
    Float mfw_success_ratio = Float(stats_mfw_succeeded) / (stats_mfw_succeeded + stats_mfw_failed),
          mfw_fail_ratio    = Float(stats_mfw_failed)    / (stats_mfw_succeeded + stats_mfw_failed);

    std::cout << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "    Specular Manifold Sampling Statistics" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << std::setw(25) << std::left << "walks succeeded: "
              << std::setw(10) << std::right << stats_mfw_succeeded << " "
              << std::setw(8) << "(" << 100*mfw_success_ratio << "%)" << std::endl;
    std::cout << std::setw(25) << std::left << "walks failed: "
              << std::setw(10) << std::right << stats_mfw_failed << " "
              << std::setw(8) << "(" << 100*mfw_fail_ratio << "%)" << std::endl;
    std::cout << std::endl;

    Float stats_booth_avg_iterations = Float(stats_bernoulli_trial_iterations) / stats_bernoulli_trial_calls;
    std::cout << std::setw(25) << std::left << "avg. Booth iterations: "
              << std::setw(10) << std::right << stats_booth_avg_iterations << std::endl;
    std::cout << std::setw(25) << std::left << "max. Booth iterations: "
              << std::setw(10) << std::right << stats_bernoulli_trial_iterations_max << std::endl;
    std::cout << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << std::endl;
}

MTS_IMPLEMENT_CLASS_VARIANT(SpecularManifoldMultiScatter, Object, "manifold_ms")
MTS_INSTANTIATE_CLASS(SpecularManifoldMultiScatter)
NAMESPACE_END(mitsuba)

#include <mitsuba/render/manifold.h>
#include <mitsuba/render/manifold_ss.h>
#include <mitsuba/render/manifold_ms.h>
#include <mitsuba/render/manifold_glints.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(ManifoldVertex) {
    MTS_PY_IMPORT_TYPES()
    using ManifoldVertex = ManifoldVertex<Float, Spectrum>;

    MTS_PY_STRUCT(ManifoldVertex)
        .def(py::init<const Point3f &>())
        .def(py::init<const SurfaceInteraction3f &, Float>())
        .def_readwrite("p", &ManifoldVertex::p)
        .def_readwrite("dp_du", &ManifoldVertex::dp_du)
        .def_readwrite("dp_dv", &ManifoldVertex::dp_dv)
        .def_readwrite("n", &ManifoldVertex::n)
        .def_readwrite("gn", &ManifoldVertex::gn)
        .def_readwrite("dn_du", &ManifoldVertex::dn_du)
        .def_readwrite("dn_dv", &ManifoldVertex::dn_dv)
        .def_readwrite("s", &ManifoldVertex::s)
        .def_readwrite("t", &ManifoldVertex::t)
        .def_readwrite("ds_du", &ManifoldVertex::ds_du)
        .def_readwrite("ds_dv", &ManifoldVertex::ds_dv)
        .def_readwrite("dt_du", &ManifoldVertex::dt_du)
        .def_readwrite("dt_dv", &ManifoldVertex::dt_dv)
        .def_readwrite("eta", &ManifoldVertex::eta)
        .def_readonly("shape", &ManifoldVertex::shape)
        .def_readwrite("uv", &ManifoldVertex::uv)
        .def_readwrite("C", &ManifoldVertex::C)
        .def_readwrite("dC_dx_prev", &ManifoldVertex::dC_dx_prev)
        .def_readwrite("dC_dx_cur", &ManifoldVertex::dC_dx_cur)
        .def_readwrite("dC_dx_next", &ManifoldVertex::dC_dx_next)
        .def_readwrite("inv_lambda", &ManifoldVertex::inv_lambda)
        .def_readwrite("dx", &ManifoldVertex::dx)
        .def("__repr__", &ManifoldVertex::to_string);
}

MTS_PY_EXPORT(EmitterInteraction) {
    MTS_PY_IMPORT_TYPES()
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;

    MTS_PY_STRUCT(EmitterInteraction)
        .def(py::init<>())
        .def_readwrite("p", &EmitterInteraction::p)
        .def_readwrite("n", &EmitterInteraction::n)
        .def_readwrite("d", &EmitterInteraction::d)
        .def_readwrite("pdf", &EmitterInteraction::pdf)
        .def_readonly("emitter", &EmitterInteraction::emitter)
        .def("__repr__", &EmitterInteraction::to_string);
}

MTS_PY_EXPORT(SpecularManifold) {
    MTS_PY_IMPORT_TYPES()
    using SpecularManifold = SpecularManifold<Float, Spectrum>;

    MTS_PY_STRUCT(SpecularManifold)
        .def_static("sample_emitter_interaction", &SpecularManifold::sample_emitter_interaction,
            "si"_a, "emitters"_a, "sampler"_a)
        .def_static("emitter_interaction_to_vertex", &SpecularManifold::emitter_interaction_to_vertex,
            "scene"_a, "y"_a, "v"_a, "time"_a, "wavelengths"_a)

        .def_static("sample_gaussian", &SpecularManifold::sample_gaussian,
            "mu"_a, "sigma"_a, "sample"_a);
}

MTS_PY_EXPORT(SpecularManifoldSingleScatter) {
    MTS_PY_IMPORT_TYPES()
    using SpecularManifoldSingleScatter = SpecularManifoldSingleScatter<Float, Spectrum>;

    MTS_PY_STRUCT(SpecularManifoldSingleScatter)
        .def(py::init<const Scene *, const SMSConfig &>(),
            "scene"_a, "config"_a)
        .def("specular_manifold_sampling", &SpecularManifoldSingleScatter::specular_manifold_sampling,
             "si"_a, "sampler"_a)
        .def("sample_path", &SpecularManifoldSingleScatter::sample_path,
            "shape"_a, "si"_a, "ei"_a, "sampler"_a,
            "n_offset"_a=Vector3f(0.f, 0.f, 1.f), "p_start"_a=Point3f(0.f))
        .def("newton_solver", &SpecularManifoldSingleScatter::newton_solver,
             "si_p"_a, "vtx_init"_a, "ei"_a, "offset"_a=Vector2f(0.f), "smoothing"_a=0.f)
        .def("compute_step_halfvector", &SpecularManifoldSingleScatter::compute_step_halfvector,
            "v0p"_a, "v1"_a, "v2p"_a, "n_offset"_a=Vector3f(0.f, 0.f, 1.f))
        .def("compute_step_anglediff", &SpecularManifoldSingleScatter::compute_step_anglediff,
            "v0p"_a, "v1"_a, "v2p"_a, "n_offset"_a=Vector3f(0.f, 0.f, 1.f));
}

MTS_PY_EXPORT(SpecularManifoldGlints) {
    MTS_PY_IMPORT_TYPES()
    using SpecularManifoldGlints = SpecularManifoldGlints<Float, Spectrum>;

    MTS_PY_STRUCT(SpecularManifoldGlints)
        .def(py::init<const Scene *, const SMSConfig &>(),
            "scene"_a, "config"_a)
        .def("specular_manifold_sampling", &SpecularManifoldGlints::specular_manifold_sampling,
             "sensor_position"_a, "si"_a, "sampler"_a)
        .def("sample_glint", &SpecularManifoldGlints::sample_glint,
            "sensor_position"_a, "ei"_a, "si"_a, "sampler"_a,
            "n_offset"_a=Vector3f(0.f, 0.f, 1.f), "xi_start"_a=Point2f(1.f));
}
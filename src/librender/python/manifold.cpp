#include <mitsuba/render/manifold.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(SMSConfig) {
     MTS_PY_STRUCT(SMSConfig)
        .def(py::init<>())
        .def_readwrite("biased", &SMSConfig::biased)
        .def_readwrite("twostage", &SMSConfig::twostage)
        .def_readwrite("halfvector_constraints", &SMSConfig::halfvector_constraints)
        .def_readwrite("mnee_init", &SMSConfig::mnee_init)
        .def_readwrite("step_scale", &SMSConfig::step_scale)
        .def_readwrite("max_iterations", &SMSConfig::max_iterations)
        .def_readwrite("solver_threshold", &SMSConfig::solver_threshold)
        .def_readwrite("uniqueness_threshold", &SMSConfig::uniqueness_threshold)
        .def_readwrite("max_trials", &SMSConfig::max_trials)
        .def("__repr__", &SMSConfig::to_string);
}

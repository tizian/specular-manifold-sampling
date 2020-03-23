#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/endpoint.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT Emitter<Float, Spectrum>::Emitter(const Properties &props) : Base(props) {
    m_caustic_emitter_single = props.bool_("caustic_emitter_single", false);
    m_caustic_emitter_multi  = props.bool_("caustic_emitter_multi", false);
}

MTS_VARIANT Emitter<Float, Spectrum>::~Emitter() { }

MTS_IMPLEMENT_CLASS_VARIANT(Emitter, Endpoint, "emitter")
MTS_INSTANTIATE_CLASS(Emitter)
NAMESPACE_END(mitsuba)

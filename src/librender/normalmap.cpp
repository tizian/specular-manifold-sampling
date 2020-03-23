#include <mitsuba/render/normalmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/plugin.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT Normalmap<Float, Spectrum>::Normalmap(const std::string &filename) {
    m_filename = filename;

    ref<Bitmap> normalmap;
    fs::path file_path(filename);
    if (file_path.extension() == ".png" ||
        file_path.extension() == ".jpg") {
        Log(Warn, "Normalmap: using an 8-bit format is discouraged due to numerical imprecision when computing normalmap derivatives for SMS.");
        normalmap = new Bitmap(file_path);
        normalmap = normalmap->convert(Bitmap::PixelFormat::RGB, struct_type_v<ScalarFloat>, true);
    } else if (file_path.extension() == ".exr") {
        normalmap = new Bitmap(file_path);
        normalmap = normalmap->convert(Bitmap::PixelFormat::RGB, struct_type_v<ScalarFloat>, false);
    } else {
        Throw("Normalmap(): Unkown file format.");
    }

    size_t width = normalmap->size().x(),
           height = normalmap->size().y();

    if (!(math::is_power_of_two(width) &&
          math::is_power_of_two(width) &&
          width == height)) {
        Throw("Expect \"square power-of-two\" resolution normal map.");
    }

    size_t size = width;
    m_sizes.push_back(size);

    // Base roughness for LEAN map
    Float base_alpha = 0.0001f;     // Beckmann roughness
    Float base_sigma_2 = 0.5f*base_alpha*base_alpha;

    // Setup first level of LEAN map
    ref<Bitmap> leanmap = new Bitmap(Bitmap::PixelFormat::MultiChannel, struct_type_v<ScalarFloat>, ScalarVector2u(size), 5);
    ScalarFloat *src = (ScalarFloat *) normalmap->data();
    ScalarFloat *dst = (ScalarFloat *) leanmap->data();
    for (size_t s = 0; s < size*size; ++s) {
        ScalarColor3f rgb = load_unaligned<ScalarColor3f>(src);

        ScalarFloat tmp1 = 2*rgb[0] - 1;
        ScalarFloat tmp2 = 2*rgb[1] - 1;
        ScalarFloat tmp3 = 2*rgb[2] - 1;
        ScalarNormal3f n = normalize(ScalarNormal3f(tmp1, tmp2, tmp3));

        ScalarFloat slope_x = -n[0] / n[2];
        ScalarFloat slope_y = -n[1] / n[2];
        ScalarFloat slope_xx = slope_x*slope_x + base_sigma_2;
        ScalarFloat slope_yy = slope_y*slope_y + base_sigma_2;
        ScalarFloat slope_xy = slope_x*slope_y;

        store_unaligned<ScalarVector2f>(dst, ScalarVector2f(slope_x, slope_y));
        dst += 2;
        store_unaligned<ScalarVector3f>(dst, ScalarVector3f(slope_xx, slope_yy, slope_xy));
        dst += 3;

        src += 3;
    }
    m_normals_mipmap.push_back(normalmap);
    m_lean_mipmap.push_back(leanmap);

    // Resample remaining mipmap levels
    ref<ReconstructionFilter> rfilter =
        PluginManager::instance()->create_object<ReconstructionFilter>(
            Properties("box")
        );
    while (size > 1) {
        size = size / 2;
        m_sizes.push_back(size);

        ref<Bitmap> normalmap_downsampled = normalmap->resample(ScalarVector2u(size), rfilter);
        ref<Bitmap> leanmap_downsampled = leanmap->resample(ScalarVector2u(size), rfilter);

        m_normals_mipmap.push_back(normalmap_downsampled);
        m_lean_mipmap.push_back(leanmap_downsampled);

        normalmap = normalmap_downsampled;
        leanmap = leanmap_downsampled;
    }
}

MTS_VARIANT Normalmap<Float, Spectrum>::~Normalmap() {}

MTS_IMPLEMENT_CLASS_VARIANT(Normalmap, Object, "normalmap_data")
MTS_INSTANTIATE_CLASS(Normalmap)
NAMESPACE_END(mitsuba)

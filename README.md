<img src="https://github.com/tizian/specular-manifold-sampling/raw/master/docs/images/sms-teaser.jpg" alt="SMS teaser">

# Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints

Source code of the paper ["Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints"](https://rgl.epfl.ch) by [Tizian Zeltner](https://tizianzeltner.com/), [Iliyan Georgiev](http://www.iliyan.com/), and [Wenzel Jakob](http://rgl.epfl.ch/people/wjakob) from SIGGRAPH 2020.

The implementation is based on the Mitsuba 2 Renderer, see the lower part of the README.

## Compilation

The normal compilation instructions for Mitsuba 2 apply. See the ["Getting started"](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/intro.html) sections in the documentation. For this project, only the *scalar_{rgb,spectral}* variants are tested. The paper shows results generated with *scalar_rgb*.

## Overview

Various versions of the SMS technique are implemented:

**Single-scattering caustic SMS**

Sampling technique for diffuse-specular-light connections with a single reflection or refraction event.

* Implemented in `include/mitsuba/render/manifold_ss.h` and `src/librender/manifold_ss.cpp`
* Augmented path tracer in `src/integrators/path_sms_ss.cpp`
* Reference path tracer (that renders the same subset of light paths for comparisons) in `src/integrators/path_filtered_ss.cpp`
* Used to render Figures 4, 6, 8, 9, 14, 15, 16, 17 in the paper.

**Multi-scattering caustic SMS**

Sampling technique for diffuse-specular*-light connections with a fixed number of reflection or refraction events.

* Implemented in `include/mitsuba/render/manifold_ms.h` and `src/librender/manifold_ms.cpp`
* Augmented path tracer in `src/integrators/path_sms_ms.cpp`
* Reference path tracer (that renders the same subset of light paths for comparisons) in `src/integrators/path_filtered_ms.cpp`
* Used to render Figure 18 in the paper.

**Glint SMS**

Sampling technique for glints from specular (normal-mapped) microstructures.

* Implemented in `include/mitsuba/render/manifold_glints.h` `src/librender/manifold_glints.cpp`
* Augmented path tracer in `src/integrators/path_sms_ms.cpp`
* Used to render Figures 12 and 19 in the paper.

**Vectorized Glint SMS**

Since the submission, we also implemented a version of the glints that use of *SIMD vectorization*.

* Implemented in `include/mitsuba/render/manifold_glints_vectorized.h` `src/librender/manifold_glints_vectorized.cpp`
* Uses the same integrator as the scalar glints, but with the xml flag `<boolean name="glints_vectorized" value="true"/>` in the scene description
* Does not support surface roughness
* Only supports the *biased* inverse PDF estimate

**Combined caustic and glint integrators**

We also combined the previous single/multi-scattering caustics and the glint method into a single integrator that was used for the teaser image.

* Augmented path tracer in `src/integrators/path_sms_teaser.cpp`
* Reference path tracer (that renders the same subset of light paths for comparisons) in `src/integrators/path_filtered_teaser.cpp`
* Used to render Figures 1 and 13 in the paper.


## Results

The directory `results` contains a set of folders for the different figures in the paper, e.g. `results/Figure_<N>_<Name>`. They contain Python scripts (to generate plots or compute BSDFs) as well as Mitsuba 2 scenes for rendered results.

* All of these scripts need to be run *from the respective subfolder* to ensure that files are written to existing directories.
* Most scripts assume that Mitsuba was added to the path either manually or by running `source setpaht.sh`. See the ["Running Mitsuba"](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/compiling.html#running-mitsuba) section in the documentation.
* Note that the output for the various *equal time comparisons* will be more or less converged depending on your hardware, and will likely not match the exact renderings from the paper.

Here is a list of available results:


### `results/Figure_4_5_RingSolutions/`

* Run `mitsuba ring.xml` to render Figure 4a.
* Run Python script `generate_plots_simple.py` to create the two plots in Figure b,d.
* Run `mitsuba ring_paths.xml` to render Figure 4c.
* Run Python script `generate_fractal.py` to create Figure 5a.
* Run Python script `generate_plots_normalmapped.py` to create Figure 5b.


### `results/Figure_6_Sequence/`

* Run Python script `render.py` that renders the scene with Mitsuba after setting the right method parameters.
* Run Python script `render_references.py` to render references with path tracing and SMS. This will take a long time!


### `results/Figure_8_Constraints/`

* Run Python script `render.py` that renders the scene using the two approaches.


### `results/Figure_9_Twostage/`

* Run Python script `render.py` that renders the two scenes with both approaches.


### `results/Figure_10_TwostageSolutions/`

* Run Python script `generate_plots.py` to create the two subplots.


### `results/Figure_11_GlintsZoom/`

* Run Python script `generate_plots.py` to create plots of the footprint and the convergence basins inside.


### `results/Figure_12_GlintsMIS/`

* Run Python script `render.py` that renders the three images with Mitsuba after setting the right method parameters.


### `results/Figure_14_15_MainComparison/`

* Run Python scripts `render_{plane,sphere,pool}.py` to create renderings for all methods at equal time.
* Run Python scripts `render_references_{plane,sphere,pool}.py` to render references with path tracing and SMS. This will take a long time!


### `results/Figure_16_Displacement/`

* Run Python script `render.py` to render both versions of the scene.


### `results/Figure_17_Roughness/`

* Run Python script `render.py` to render the scenes with varying roughness using both approaches.


### `results/Figure_18_DoubleRefraction/`

* Run Python script `render.py` to create renderings for all methods at equal time.
* Run Python script `render_references.py` to render references with path tracing and SMS. This will take a long time!


### `results/Figure_19_GlintsComparison/`

* Run Python script `generate_normalmaps.py` that will create the high-resolution normal maps used in the two scenes.
* In order to run prior work ["Position-Normal Distributions for Efficient Rendering of Specular Microstructure" by Yan et al. 2016](https://sites.cs.ucsb.edu/~lingqi/publications/paper_glints2.pdf), convert the normal maps to the `.flakes` format used by their method by running these two commands:
   * `./<Mitsuba 2 build directory>/dist/normalmap_to_flakes textures/normalmap_gaussian_yan.exr gaussian.flakes 4`
   * `./<Mitsuba 2 build directory>/dist/normalmap_to_flakes textures/normalmap_brushed_yan.exr brushed.flakes 2`
* Render sequences of renderings with increasing time by running the `render_{shoes,kettle}.py` scripts. These run for a long time! Specify the method to use by providing one of the following command line arguments:
   * `pt` for path tracer reference
   * `sms_ub` for unbiased SMS
   * `sms_b` for biased SMS
   * `sms_bv` for biased + vectorized SMS
   * `yan` for the method of Yan et al. 2016
* Render path traced reference insets with Python script `render_references.py`.
* Process the renderings and log files to generate the convergence plots with Python script `generate_plots.py`.

---

<img src="https://github.com/mitsuba-renderer/mitsuba2/raw/master/docs/images/logo_plain.png" width="120" height="120" alt="Mitsuba logo">

# Mitsuba Renderer 2
<!--
| Documentation   | Linux             | Windows             |
|      :---:      |       :---:       |        :---:        |
| [![docs][1]][2] | [![rgl-ci][3]][4] | [![appveyor][5]][6] |


[1]: https://readthedocs.org/projects/mitsuba2/badge/?version=master
[2]: https://mitsuba2.readthedocs.io/en/latest/src/getting_started/intro.html
[3]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:Mitsuba2_Build)/statusIcon.svg
[4]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=Mitsuba2_Build&guest=1
[5]: https://ci.appveyor.com/api/projects/status/eb84mmtvnt8ko8bh/branch/master?svg=true
[6]: https://ci.appveyor.com/project/wjakob/mitsuba2/branch/master
-->
| Documentation   |
|      :---:      |
| [![docs][1]][2] |


[1]: https://readthedocs.org/projects/mitsuba2/badge/?version=latest
[2]: https://mitsuba2.readthedocs.io/en/latest/src/getting_started/intro.html

Mitsuba 2 is a research-oriented rendering system written in portable C++17. It
consists of a small set of core libraries and a wide variety of plugins that
implement functionality ranging from materials and light sources to complete
rendering algorithms. Mitsuba 2 strives to retain scene compatibility with its
predecessor [Mitsuba 0.6](https://github.com/mitsuba-renderer/mitsuba).
However, in most other respects, it is a completely new system following a
different set of goals.

The most significant change of Mitsuba 2 is that it is a *retargetable*
renderer: this means that the underlying implementations and data structures
are specified in a generic fashion that can be transformed to accomplish a
number of different tasks. For example:

1. In the simplest case, Mitsuba 2 is an ordinary CPU-based RGB renderer that
   processes one ray at a time similar to its predecessor [Mitsuba
   0.6](https://github.com/mitsuba-renderer/mitsuba).

2. Alternatively, Mitsuba 2 can be transformed into a differentiable renderer
   that runs on NVIDIA RTX GPUs. A differentiable rendering algorithm is able
   to compute derivatives of the entire simulation with respect to input
   parameters such as camera pose, geometry, BSDFs, textures, and volumes. In
   conjunction with gradient-based optimization, this opens door to challenging
   inverse problems including computational material design and scene reconstruction.

3. Another type of transformation turns Mitsuba 2 into a vectorized CPU
   renderer that leverages Single Instruction/Multiple Data (SIMD) instruction
   sets such as AVX512 on modern CPUs to efficiently sample many light paths in
   parallel.

4. Yet another type of transformation rewrites physical aspects of the
   simulation: Mitsuba can be used as a monochromatic renderer, RGB-based
   renderer, or spectral renderer. Each variant can optionally account for the
   effects of polarization if desired.

In addition to the above transformations, there are
several other noteworthy changes:

1. Mitsuba 2 provides very fine-grained Python bindings to essentially every
   function using [pybind11](https://github.com/pybind/pybind11). This makes it
   possible to import the renderer into a Jupyter notebook and develop new
   algorithms interactively while visualizing their behavior using plots.

2. The renderer includes a large automated test suite written in Python, and
   its development relies on several continuous integration servers that
   compile and test new commits on different operating systems using various
   compilation settings (e.g. debug/release builds, single/double precision,
   etc). Manually checking that external contributions don't break existing
   functionality had become a severe bottleneck in the previous Mitsuba 0.6
   codebase, hence the goal of this infrastructure is to avoid such manual
   checks and streamline interactions with the community (Pull Requests, etc.)
   in the future.

3. An all-new cross-platform user interface is currently being developed using
   the [NanoGUI](https://github.com/mitsuba-renderer/nanogui) library. *Note
   that this is not yet complete.*

## Compiling and using Mitsuba 2

Please see the [documentation](http://mitsuba2.readthedocs.org/en/latest) for
details on how to compile, use, and extend Mitsuba 2.

## About

This project was created by [Wenzel Jakob](http://rgl.epfl.ch/people/wjakob).
Significant features and/or improvements to the code were contributed by
[Merlin Nimier-David](https://merlin.nimierdavid.fr/),
[Guillaume Loubet](https://maverick.inria.fr/Membres/Guillaume.Loubet/),
[Sébastien Speierer](https://github.com/Speierers),
[Delio Vicini](https://dvicini.github.io/),
and [Tizian Zeltner](https://tizianzeltner.com/).
try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import Bitmap, Struct

import numpy as np


# Based on the tech report "Generating Procedural Beckmann Surfaces" by Eric Heitz
# https://drive.google.com/file/d/0BzvWIdpUpRx_U1NOUjlINmljQzg/view

def gauss_normalmap(alpha_x, alpha_y, res, scale, num_waves=1000, name=None, noise=0.0, encoding_yan16=False, seed=1):
    np.random.seed(seed)
    U1 = np.random.rand(num_waves)
    U2 = np.random.rand(num_waves)
    U3 = np.random.rand(num_waves)

    factor = np.sqrt(2 / num_waves)

    phi   = 2*np.pi*U1
    theta = 2*np.pi*U2
    r = np.sqrt(-np.log(1.0 - U3))

    fx = r*np.cos(theta) * alpha_x
    fy = r*np.sin(theta) * alpha_y

    def height(pos):
        return factor*np.sum(np.cos(phi + pos[0]*fx + pos[1]*fy))

    def slope(pos):
        tmp = np.sin(phi + pos[0]*fx + pos[1]*fy)
        res_x = np.sum(-fx*tmp)
        res_y = np.sum(-fy*tmp)
        return factor*np.array([res_x, res_y])

    def normal(pos, noise):
        s = slope(pos)
        n = np.array([-s[0] + noise*np.random.uniform(-1, 1),
                      -s[1] + noise*np.random.uniform(-1, 1),
                      1.0])
        return n / np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)

    xx = np.linspace(0, scale, res)
    yy = np.linspace(0, scale, res)

    normalmap = np.zeros((res, res, 3))
    for j, x in enumerate(xx):
        print ("Generate \"%s\": %.2f%%" % (name, 100*j/(res-1)), end="\r")
        for i, y in enumerate(yy):
            n = normal(np.array([x, y]), noise)
            if encoding_yan16:
                normalmap[i, j, 0] = n[0]
                normalmap[i, j, 1] = n[1]
                normalmap[i, j, 2] = n[2]
            else:
                normalmap[i, j, 0] = 0.5*(n[0] + 1.0)
                normalmap[i, j, 1] = 0.5*(n[1] + 1.0)
                normalmap[i, j, 2] = 0.5*(n[2] + 1.0)
    Bitmap(normalmap).convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, False).write("textures/{}.exr".format(name))
    print("")

gauss_normalmap(0.6, 0.6, 2048, 1000, name='normalmap_gaussian', seed=1)
gauss_normalmap(0.6, 0.6, 512, 1000, name='normalmap_gaussian_yan', seed=1, encoding_yan16=True)

gauss_normalmap(0.001, 0.8, 2048, 10000, name='normalmap_brushed', seed=3, noise=0.04)
gauss_normalmap(0.001, 0.8, 2048, 10000, name='normalmap_brushed_yan', seed=3, noise=0.04, encoding_yan16=True)
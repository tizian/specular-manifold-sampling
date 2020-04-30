try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.render import SMSConfig, SpecularManifold, SpecularManifoldGlints
mitsuba.core.Thread.thread().logger().set_log_level(mitsuba.core.LogLevel.Error)

import enoki as ek
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

print("Load glinty scene ..")
scene_path = "scene_zoom.xml"
scene       = mitsuba.core.xml.load_file(scene_path)
camera      = scene.sensors()[0]
integrator  = scene.integrator()

res = camera.film().size()
inv_res = ek.rcp(ek.scalar.Vector2f(res))

sampler = mitsuba.core.xml.load_string("<sampler version='2.0.0' type='independent'/>")


np.random.seed(0)
sampler.seed(0)

glint_mask = np.zeros((res[1], res[0]))

for j in range(res[0]):
    print ("Compute glint mask: %.2f%%" % (100*j/(res[0]-1)), end="\r")
    for i in range(res[1]):
        p = np.array([j, i]) * inv_res

        ray, _ = camera.sample_ray_differential(0.0, 0.5, p, [0.5, 0.5])
        si = scene.ray_intersect(ray)
        if si.shape.is_glinty():
            glint_mask[i,j] = 1
        else:
            glint_mask[i,j] = 0
print("")

# plt.imshow(glint_mask)
# plt.show()


np.random.seed(0)
sampler.seed(0)

glinty_pixels = []

sms_config = SMSConfig()
sms_config.max_iterations = 20
sms_config.solver_threshold = 1e-5
mf = SpecularManifoldGlints()
mf.init(scene, sms_config)

for j in range(res[0]):
    print ("Identify glinty pixels: %.2f%%" % (100*j/(res[0]-1)), end="\r")
    for i in range(res[1]):
        if glint_mask[i,j] == 0:
            continue
        p = np.array([j+0.5, i+0.5]) * inv_res

        ray, _ = camera.sample_ray_differential(0.0, 0.5, p, [0.5, 0.5])
        si = scene.ray_intersect(ray)
        si.compute_partials(ray)

        ei = SpecularManifold.sample_emitter_interaction(si, scene.emitters(), sampler)

        wi = ek.normalize(ray.o - si.p)
        wo = ek.normalize(ei.p  - si.p)
        h = ek.normalize(wi + wo)

        n_offset = np.array([0,0,1])
        for k in range(1):
            success, uv_final, _, = mf.sample_glint(ray.o, ei, si, sampler, n_offset)
            if success:
                glinty_pixels.append((i,j))
                break
print("")

# print("Found {} glinty pixels".format(len(glinty_pixels)))

idx = 238 # Manually chosen index that gives nice looking results

tmp = np.copy(glint_mask)
highlight_mask = np.dstack([tmp, tmp, tmp])
pixel_ij = glinty_pixels[idx]

sampler.seed(0)
np.random.seed(0)

ray = None
si = None
ei = None

done = False
for j in range(res[0]):
    print ("Find solutions in footprint: %.2f%%" % (100*j/(res[0]-1)), end="\r")
    if done:
        j = res[0]-1
        print ("Find solutions in footprint: %.2f%%" % (100*j/(res[0]-1)), end="\r")
        break
    for i in range(res[1]):
        if glint_mask[i,j] == 0:
            continue
        p = np.array([j+0.5, i+0.5]) * inv_res

        ray, _ = camera.sample_ray_differential(0.0, 0.5, p, [0.5, 0.5])
        si = scene.ray_intersect(ray)
        si.compute_partials(ray)

        ei = SpecularManifold.sample_emitter_interaction(si, scene.emitters(), sampler)

        n_offset = np.array([0,0,1])
        success, uv_final, uv_init, = mf.sample_glint(ray.o, ei, si, sampler, n_offset)

        if i != pixel_ij[0] or j != pixel_ij[1]:
            continue

        N = 30
        uvs = np.linspace(0, 1, N)

        seed_uv = np.zeros((N*N, 2))
        solutions_uv = []

        k = -1
        for u in uvs:
            for v in uvs:
                k += 1
                success, uv_final, uv_init, = mf.sample_glint(ray.o,
                                                              ei, si,
                                                              sampler,
                                                              n_offset,
                                                              np.array([u,v]))

                seed_uv[k,:] = uv_init
                if not success:
                    continue

                duplicate = False
                for s_uv in solutions_uv:
                    if np.linalg.norm(uv_final - s_uv) < 1e-5:
                        duplicate = True
                        break
                if duplicate:
                    continue
                solutions_uv.append(uv_final)
        done = True
        break
print("")

solutions_uv = np.array(solutions_uv)
# print(len(solutions_uv), "solutions")


plt.figure(figsize=(10,10))
k = -1
for u in uvs:
    for v in uvs:
        k += 1
        plt.plot(seed_uv[k,0], seed_uv[k,1], 'o', color=np.array([u,v,0]), ms=3)

for s_uv in solutions_uv:
    plt.plot(s_uv[0], s_uv[1], 'co', ms=5, mec='k')

plt.axis('square')
plt.xticks([]); plt.yticks([])
# plt.show()
plt.savefig("glint_footprint.png",
            transparent=False, bbox_inches='tight', pad_inches=0, dpi=150)



colors = np.zeros((max(4, len(solutions_uv)),3))
colors[0,:] = np.array([1.0, 0.325, 0.349])
colors[1,:] = np.array([0.408, 0.871, 1.0])
colors[2,:] = np.array([0.482, 0.925, 0.451])
colors[3,:] = np.array([0.937, 0.867, 0.267])

res_uv = 250
uv = np.linspace(0, 1, res_uv)
map_color = np.zeros((res_uv, res_uv, 3))
map_idx = np.zeros((res_uv, res_uv))
map_idx[:,:] = -1

for j, u in enumerate(uv):
    print ("Compute convergence basin plot: %.2f%%" % (100*j/(res_uv-1)), end="\r")
    for i, v in enumerate(uv):
        success, uv_final, uv_init, = mf.sample_glint(ray.o,
                                                      ei, si,
                                                      sampler,
                                                      n_offset,
                                                      np.array([u,v]))

        if not success:
            continue
        else:
            idx = np.argmin(np.linalg.norm(solutions_uv - np.array(uv_final), axis=1))
            map_color[i, j, :] = colors[idx]
            map_idx[i,j] = idx
print("")

plt.figure(figsize=(10,10))
plt.imshow(map_color, extent=[0, 1, 1, 0], interpolation='bilinear')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
# plt.show()
plt.savefig("glint_zoom.png",
            transparent=False, bbox_inches='tight', pad_inches=0, dpi=150)

print("done.")

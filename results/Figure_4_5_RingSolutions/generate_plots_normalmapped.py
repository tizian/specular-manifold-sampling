try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.render import SMSConfig, SpecularManifold, SpecularManifoldSingleScatter
mitsuba.core.Thread.thread().logger().set_log_level(mitsuba.core.LogLevel.Error)

import enoki as ek
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"



print("Load normal mapped ring scene ..")
scene_path  = "ring_notebook_normalmapped.xml"
scene       = mitsuba.core.xml.load_file(scene_path)
camera      = scene.sensors()[0]
integrator  = scene.integrator()

res = camera.film().size()
res //= 2
inv_res = ek.rcp(ek.scalar.Vector2f(res))

emitter    = scene.shapes()[0]
spec_shape = scene.shapes()[1]

sampler = mitsuba.core.xml.load_string("<sampler version='2.0.0' type='independent'/>")

sms_config = SMSConfig()
sms_config.max_iterations = 20
sms_config.solver_threshold = 1e-5
sms_config.halfvector_constraints = True
mf = SpecularManifoldSingleScatter(scene, sms_config)

seed = 1

sampler.seed(0)
np.random.seed(seed)

p = np.array([np.random.randint(res[0]),
              np.random.randint(res[1])]) * inv_res

ray, _ = camera.sample_ray_differential(0.0, 0.5, p, [0.5, 0.5])
si = scene.ray_intersect(ray)
ei = SpecularManifold.sample_emitter_interaction(si, scene.emitters(), sampler)

solutions = []
solutions_uv = []
solutions_p = []
n_tries = 500
for tries in range(n_tries):
    print ("Find solutions: %.2f%%" % (100*tries/(n_tries-1)), end="\r")
    success, si_final, _ = mf.sample_path(spec_shape, si, ei, sampler)
    if not success:
        continue
    direction = ek.normalize(si_final.p - si.p)

    duplicate = False
    for s in solutions:
        if np.abs((direction @ s) - 1) < 1e-5:
            duplicate = True
            break
    if duplicate:
        continue

    solutions.append(direction)
    solutions_uv.append(si_final.uv)
    solutions_p.append(si_final.p)
print("")

# print("Found {} solutions.".format(len(solutions)))
# for i in solutions_p:
#     print("{:0.3f}, {:0.3f}, {:0.3f}".format(*i))

solutions_uv = np.array(solutions_uv)
solutions = np.array(solutions)


res_uv = 800
uv = np.linspace(0, 1, res_uv)
map_idx = np.zeros((res_uv, res_uv))
for j, u in enumerate(uv):
    print ("Compute convergence basin plot: %.2f%%" % (100*j/(res_uv-1)), end="\r")
    for i, v in enumerate(uv):
        p_init = spec_shape.sample_position(0.0, [v, u])
        success, si_final, _ = mf.sample_path(spec_shape, si, ei, sampler, p_start=p_init.p)
        direction = ek.normalize(si_final.p - si.p)
        if not success:
            map_idx[i,j] = -1
        else:
            idx = np.argmin(np.linalg.norm(solutions - direction, axis=1))
            map_idx[i,j] = idx
print("")

colors = np.zeros((len(solutions), 3))
colors[0,:] = np.array([0.408, 0.871, 1.000])
colors[1,:] = np.array([0.482, 0.925, 0.451])
colors[2,:] = np.array([0.894, 0.451, 0.922])
colors[3,:] = np.array([0.913, 0.922, 0.275])
colors[4,:] = np.array([1.000, 0.325, 0.349])

map_color = np.ones((res_uv, res_uv, 3))*0.0
for j, u in enumerate(uv):
    for i, v in enumerate(uv):
        idx = int(map_idx[i,j])
        if idx >= 0:
            map_color[i, j, :] = colors[idx,:]

plt.figure(figsize=(8,8))
plt.imshow(map_color, extent=[0, 1, 1, 0], interpolation='bilinear')
for i, s in enumerate(solutions_uv):
    plt.plot(s[0], s[1], 'o', ms=8, color='k')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
# plt.show()
plt.savefig("solutions_map_normalmapped.png",
            transparent=False, bbox_inches='tight', pad_inches=0)


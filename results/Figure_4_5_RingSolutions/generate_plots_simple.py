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

# Adapted from https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
def discrete_matshow(data, ax):
    # Get nice colors
    cmap = plt.get_cmap('tab20c')
    color_table = cmap(np.linspace(0,1,20))

    # Use correct scale, but replace all colors manually :)
    cmap = plt.get_cmap('tab20c', int(np.max(data)-np.min(data)+1))
    colors = cmap(np.linspace(0, 1, 4))
    colors[3,:] = color_table[0+0,:]
    colors[2,:] = color_table[0+1,:]
    colors[1,:] = color_table[0+2,:]
    colors[0,:] = color_table[17,:]
    newcmp = ListedColormap(colors)

    mat = ax.matshow(data, cmap=newcmp,
                     vmin = np.min(data)-.5, vmax = np.max(data)+.5,
                     interpolation='bilinear')

    cax = plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1), ax=ax, fraction=0.0353, pad=0.02)
    cax.ax.tick_params(labelsize=18)

print("Load ring scene ..")
scene_path = "ring_notebook_simple.xml"
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
mf = SpecularManifoldSingleScatter(scene, sms_config)

sampler.seed(0)
ray, _ = camera.sample_ray_differential(0.0, 0.5, [0.5, 0.5], [0.5, 0.5])
si = scene.ray_intersect(ray)
ei = SpecularManifold.sample_emitter_interaction(si, scene.emitters(), sampler)

number_solutions = np.zeros((res[1], res[0]))
for j in range(res[0]):
    print ("Compute plot of solutions numbers: %.2f%%" % (100*j/(res[0]-1)), end="\r")
    for i in range(res[1]):
        p = np.array([j, i]) * inv_res
        ray, _ = camera.sample_ray_differential(0.0, 0.5, p, [0.5, 0.5])
        si = scene.ray_intersect(ray)
        if si.shape == spec_shape:
            continue

        solutions = []
        for tries in range(50):
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

        number_solutions[i,j] = len(solutions)

fig, ax = plt.subplots(figsize=(10,10))
discrete_matshow(number_solutions, ax)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
# plt.show()
plt.savefig("number_solutions.png", transparent=False, bbox_inches='tight', pad_inches=0)
print("")


seed = 38           # Manually chosen seed that gives nice looking results
sampler.seed(0)
np.random.seed(seed)

p = np.array([np.random.randint(res[0]),
              np.random.randint(res[1])]) * inv_res

ray, _ = camera.sample_ray_differential(0.0, 0.5, p, [0.5, 0.5])
si = scene.ray_intersect(ray)

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

colors = np.zeros((3,3))
colors[0,:] = np.array([1.0, 0.325, 0.349])
colors[1,:] = np.array([0.408, 0.871, 1.0])
colors[2,:] = np.array([0.482, 0.925, 0.451])

sms_config = SMSConfig()
sms_config.max_iterations = 20
sms_config.solver_threshold = 1e-5
sms_config.halfvector_constraints = True
mf = SpecularManifoldSingleScatter(scene, sms_config)

res_uv = 150
uv = np.linspace(0, 1, res_uv)
map_color = np.ones((res_uv, res_uv, 3))*0.0
map_idx = np.zeros((res_uv, res_uv))
for j, u in enumerate(uv):
    print ("Compute convergence basin plot: %.2f%%" % (100*j/(res_uv-1)), end="\r")
    for i, v in enumerate(uv):
        p_init = spec_shape.sample_position(0.0, [v, u])
        success, si_final, _ = mf.sample_path(spec_shape, si, ei, sampler, p_start=p_init.p)
        direction = ek.normalize(si_final.p - si.p)
        if not success:
            map_color[i, j, :] = 0.0
            map_idx[i,j] = -1
        else:
            idx = np.argmin(np.linalg.norm(solutions - direction, axis=1))
            map_color[i, j, :] = colors[idx]
            map_idx[i,j] = idx
print("")

plt.figure(figsize=(8,8))
plt.imshow(map_color, extent=[0, 1, 1, 0], interpolation='bilinear')
for i, s in enumerate(solutions_uv):
    plt.plot(s[0], s[1], 'o', ms=8, color='k')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
# plt.show()
plt.savefig("solutions_map.png",
            transparent=False, bbox_inches='tight', pad_inches=0)

print("done.")

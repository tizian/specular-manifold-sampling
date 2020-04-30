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
from matplotlib import cm
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

print("Load bumpy scene ..")
scene_path = "plane_notebook.xml"
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

seed = 35   # Manually chosen seed that gives nice looking results

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


sms_config = SMSConfig()
sms_config.max_iterations = 20
sms_config.solver_threshold = 1e-5
sms_config.halfvector_constraints = False
mf = SpecularManifoldSingleScatter(scene, sms_config)

res_uv = 300
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
for ii, xx in enumerate(np.linspace(0, 1, len(solutions), endpoint=False)):
    tmp = np.array(cm.hsv(xx))[:3]
    colors[ii,:] = tmp

map_color = np.ones((res_uv, res_uv, 3))*0.0
for j, u in enumerate(uv):
    for i, v in enumerate(uv):
        idx = int(map_idx[i,j])
        if idx >= 0:
            map_color[i, j, :] = colors[idx,:]

plt.figure(figsize=(8,8))
plt.imshow(map_color, extent=[0, 1, 1, 0], interpolation='bilinear')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
# plt.show()
plt.savefig("solutions_map_normalmapped.png",
            transparent=False, bbox_inches='tight', pad_inches=0)



print("Load and switch to smoothed scene ..")
scene_path = "plane_notebook_smoothed.xml"
scene_smoothed      = mitsuba.core.xml.load_file(scene_path)
emitter             = scene_smoothed.shapes()[0]
spec_shape_smoothed = scene_smoothed.shapes()[1]
mf = SpecularManifoldSingleScatter(scene_smoothed, sms_config)

ray, _ = camera.sample_ray_differential(0.0, 0.5, [0.5, 0.5], [0.5, 0.5])
si = scene.ray_intersect(ray)

si.p = [-2.942, 0.000, 5.837]

solutions = []
solutions_uv = []
solutions_p = []
n_tries = 500
for tries in range(n_tries):
    print ("Find solutions: %.2f%%" % (100*tries/(n_tries-1)), end="\r")
    success, si_final, _ = mf.sample_path(spec_shape_smoothed, si, ei, sampler)
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


sms_config = SMSConfig()
sms_config.max_iterations = 20
sms_config.solver_threshold = 1e-5
sms_config.halfvector_constraints = False
mf = SpecularManifoldSingleScatter(scene_smoothed, sms_config)

res_uv = 300
uv = np.linspace(0, 1, res_uv)
map_idx = np.zeros((res_uv, res_uv))
for j, u in enumerate(uv):
    print ("Compute convergence basin plot: %.2f%%" % (100*j/(res_uv-1)), end="\r")
    for i, v in enumerate(uv):
        p_init = spec_shape_smoothed.sample_position(0.0, [v, u])
        success, si_final, _ = mf.sample_path(spec_shape_smoothed, si, ei, sampler, p_start=p_init.p)
        direction = ek.normalize(si_final.p - si.p)
        if not success:
            map_idx[i,j] = -1
        else:
            idx = np.argmin(np.linalg.norm(solutions - direction, axis=1))
            map_idx[i,j] = idx
print("")

colors = np.zeros((len(solutions), 3))
for ii, xx in enumerate(np.linspace(0, 1, len(solutions), endpoint=False)):
    tmp = np.array(cm.hsv(xx))[:3]
    colors[ii,:] = tmp
colors[0,:] = [0.55, 0.921, 0.494]

map_color = np.ones((res_uv, res_uv, 3))*0.0
for j, u in enumerate(uv):
    for i, v in enumerate(uv):
        idx = int(map_idx[i,j])
        if idx >= 0:
            map_color[i, j, :] = colors[idx,:]

plt.figure(figsize=(8,8))
plt.imshow(map_color, extent=[0, 1, 1, 0], interpolation='bilinear')
for i, s in enumerate(solutions_uv):
    plt.plot(s[1], s[0], 'o', ms=8, color='k')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
# plt.show()
plt.savefig("solutions_map_smooth.png",
            transparent=False, bbox_inches='tight', pad_inches=0)


seedpoints = []
seedpoints_uv = []
n_samples = 1000
for tries in range(n_samples):
    print ("Sample using the first offset manifold walk: %.2f%%" % (100*tries/(n_samples-1)), end="\r")
    h = np.array([0,0,1])
    mu = np.array([-h[0]/h[2], -h[1]/h[2]])
    mu_lean, sigma_lean = spec_shape.bsdf().lean(si)

    slope = SpecularManifold.sample_gaussian(mu, sigma_lean, sampler.next_2d())
    n_lean = ek.normalize(ek.scalar.Vector3f([-slope[0], -slope[1], 1.0]))

    success, si_final, _ = mf.sample_path(spec_shape_smoothed, si, ei, sampler, n_offset=n_lean)
    if not success:
        continue
    direction = ek.normalize(si_final.p - si.p)

    duplicate = False
    for s in seedpoints:
        if np.abs((direction @ s) - 1) < 1e-5:
            duplicate = True
            break
    if duplicate:
        continue

    seedpoints.append(direction)
    seedpoints_uv.append(si_final.uv)
print("")

plt.figure(figsize=(8,8))
blank = np.ones(map_color.shape)*0.8
plt.imshow(blank, extent=[0, 1, 1, 0], interpolation='bilinear')
for i, s in enumerate(seedpoints_uv):
    plt.plot(s[1], s[0], 'o', ms=13, color='g', alpha=0.1, markeredgewidth=0)
for i, s in enumerate(solutions_uv):
    plt.plot(s[1], s[0], 'o', ms=2, color='k')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
# plt.show()
plt.savefig("twostage_seeds.png",
            transparent=False, bbox_inches='tight', pad_inches=0, dpi=150)

print("done.")

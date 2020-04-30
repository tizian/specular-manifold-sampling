import numpy as np
import matplotlib.pyplot as plt

roots = [
  np.complex(1, 0),
  np.complex(-0.5,  np.sqrt(3)/2),
  np.complex(-0.5, -np.sqrt(3)/2)
]

colors = np.zeros((3,3))
colors[0,:] = np.array([1.0, 0.325, 0.349])
colors[1,:] = np.array([0.408, 0.871, 1.0])
colors[2,:] = np.array([0.482, 0.925, 0.451])

res_x = 2000
res_y = 1500
img_colors = np.zeros((res_y, res_x, 3))
img_steps  = np.zeros((res_y, res_x, 1))
img_steps_smooth = np.zeros((res_y, res_x, 1))

max_iterations = 100
threshold      = 1e-5

for i, zy in enumerate(np.linspace(-1, 1, res_y)):
    print ("Generate fractal: %.2f%%" % (100*i/(res_y-1)), end="\r")
    for j, zx in enumerate(np.linspace(-2.0, 1.5, res_x)):
        done = False
        z = np.complex(zx, zy)

        dist = [np.inf, np.inf, np.inf]
        for it in range(max_iterations):
            f  = z**3 - 1
            df = 3*z**2
            z -= f / df

            for root_idx, root in enumerate(roots):
                d = np.abs(root - z)
                if d < threshold:
                    img_colors[i, j, :] = colors[root_idx, :]
                    img_steps[i, j, :] = it

                    tmp = np.log2(np.log(d) / np.log(threshold))
                    # Smooth shading https://www.shadertoy.com/view/3lSXz1
                    img_steps_smooth[i, j, :] = 0.75 + 0.25 * np.cos(0.25 * (it - tmp))
                    done = True
                    break
                dist[root_idx] = d

            if done:
                break

gamma = 1.0
shading = 0.0 + (img_steps_smooth[:,:]**gamma) / np.max(img_steps_smooth[:,:]**gamma)
img = img_colors*shading
img = np.clip(img, 0, 1)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.show()

plt.imsave("fractal.png", img)

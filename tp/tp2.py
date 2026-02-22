# 3D plot of z = sin(x) - cos(x) on [-2π, 2π]
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


# domain for x and y
x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
y = np.linspace(-2 * np.pi, 2 * np.pi, 200)
X, Y = np.meshgrid(x, y)

# z = sin(x) * cos(y)
Z = np.sin(X) * np.cos(Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="viridis")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="z")

ax.set_title("Surface 3D de z = sin(x) * cos(y)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()


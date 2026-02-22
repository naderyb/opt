# plot a 2D function f(x, y) = x^2 + y^2
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
	return x**2 + y**2


# create a grid of points in the (x, y) plane
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# plot the function as a filled contour map
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=40, cmap="viridis")
plt.title("2D function f(x, y) = x^2 + y^2", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.tick_params(axis="both", labelsize=10)

plt.show()


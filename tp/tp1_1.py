# 2D plot of f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
	return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


# grid of points
x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# filled contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=60, cmap="viridis")
plt.colorbar(contour, label="f(x, y)")
plt.title("2D function f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.tick_params(axis="both", labelsize=10)

plt.show()


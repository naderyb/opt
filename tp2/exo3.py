import numpy as np
import matplotlib.pyplot as plt

def rosen(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def grad_rosen(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return dx, dy

x, y = 0.2, 0.2
lr = 0.0002*10
path_x, path_y = [x], [y]

for i in range(5000):
    dx, dy = grad_rosen(x, y)
    x -= lr * dx
    y -= lr * dy
    path_x.append(x)
    path_y.append(y)

x_axis = np.linspace(-2, 2, 400)
y_axis = np.linspace(-1, 3, 400)

X, Y = np.meshgrid(x_axis, y_axis)
Z = rosen(X, Y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Rosenbrock Function Value')
plt.plot(path_x, path_y, 'ro-', label='Gradient Descent Path')
plt.title('Gradient Descent on the Rosenbrock Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
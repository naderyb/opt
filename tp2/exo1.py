import numpy as np
import matplotlib.pyplot as plt

# 1. setup
f = lambda x: x**4 - 3*x**3 + 2
df = lambda x: 4*x**3 - 9*x**2

x = 3.0 # initial guess
alpha = 0.01 # learning rate
history = [x]

# 2. execution
for _ in range(100):
    x = x - alpha * df(x) # update rule
    history.append(x)

# 3. visualization
x_axis = np.linspace(-1, 3.5, 100)
plt.plot(x_axis, f(x_axis), label='cost function')
plt.scatter(history, [f(x) for x in history], color='red', label='gradient descent path')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()
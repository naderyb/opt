import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 10*np.sin(x)

def grad_f(x):
    return 2*x + 10*np.cos(x)

def gradient_descent(initial_x, learning_rate, iterations):
    x = initial_x
    history = [x]

    for _ in range(iterations):
        x = x - learning_rate * grad_f(x)
        history.append(x)

    return np.array(history)

# simulation
initial_x = 0.0
learning_rate = 0.5
iterations = 50
history = gradient_descent(initial_x, learning_rate, iterations)

# visualization
x_axis = np.linspace(-10, 10, 400)
plt.plot(x_axis, f(x_axis), label='f(x)')
plt.scatter(history, f(history), color='red', label='gradient descent path')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()


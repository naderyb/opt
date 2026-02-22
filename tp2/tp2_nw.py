import numpy as np
import matplotlib.pyplot as plt

# Function
def f(x):
    return x**2 + 10*np.sin(x)

# First derivative
def df(x):
    return 2*x + 10*np.cos(x)

# Second derivative
def d2f(x):
    return 2 - 10*np.sin(x)

# Newton Method
def newton_method(x0, tol=1e-5, max_iter=100):
    x = x0
    history = [x]

    for i in range(max_iter):
        if abs(df(x)) < tol:
            break

        if abs(d2f(x)) < 1e-8:
            print("Hessian is near zero so Newton may fail")
            return None

        x = x - df(x)/d2f(x)
        history.append(x)

    return x, history

# Run with initial point
x0 = 2
result = newton_method(x0)

if result is not None:
    print("Minimum found at:", result[0])
    print("Number of iterations:", len(result[1]))
else:
    print("Newton method failed to converge")

# Visualization
x_vals = np.linspace(-5, 5, 400)
plt.plot(x_vals, f(x_vals))

result = newton_method(2)
if result is not None:
    _, history = result
    for x in history:
        plt.scatter(x, f(x))

plt.title("Newton Method Convergence")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
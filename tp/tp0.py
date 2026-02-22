# program that plots a 1D function
import numpy as np
import matplotlib.pyplot as plt

# define the variable x
x = np.linspace(-2, 4, 100)
# plot the function
f_x = x**3 - 3*x**2 + 2
# find the derivative
df_dx = np.gradient(f_x, x)
df2_dx2 = np.gradient(df_dx, x)
plt.plot(x, f_x)
plt.plot(x, df_dx)
plt.plot(x, df2_dx2)
plt.title("Plot of the function and its derivative")
plt.grid()
plt.xlabel("x")
plt.xlabel("y")
plt.ylabel("f(x), df/dx, d2f/dx2")
plt.legend(["f(x)", "df/dx", "d2f/dx2"])
# show the plot
plt.show()

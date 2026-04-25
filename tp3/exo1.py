import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f1(x): return (x - 1) ** 2
def f2(x): return (x - 3) ** 2
def objective(x, w): return w * f1(x) + (1 - w) * f2(x)

weights = np.linspace(0, 1, 200)
results_f1, results_f2 = [], []

for w in weights:
    res = minimize(objective, x0=0.0, args=(w,))
    if res.success:
        x_opt = res.x[0]
        results_f1.append(f1(x_opt))
        results_f2.append(f2(x_opt))

results_f1 = np.array(results_f1)
results_f2 = np.array(results_f2)

fig, ax = plt.subplots(figsize=(7, 6))

# Dotted curve: all possible x (decision space sweep)
x_all = np.linspace(-1, 5, 600)
f1_all = f1(x_all)
f2_all = f2(x_all)
ax.plot(f1_all, f2_all, linestyle=':', color='gray', linewidth=2,
    label='All possible x (decision space)')

# Solid line: Pareto front in objective space for x in [1, 3]
x_pareto = np.linspace(1, 3, 300)
f1_pareto = f1(x_pareto)
f2_pareto = f2(x_pareto)
ax.plot(f1_pareto, f2_pareto, color='black', linewidth=2.5,
    label='Pareto front (x in [1,3])')

# Red ball: min f1 at x=1
ax.scatter([f1(1)], [f2(1)], color='red', s=90, zorder=5,
       label='Min f1 (x=1)')

# Blue ball: min f2 at x=3
ax.scatter([f1(3)], [f2(3)], color='blue', s=90, zorder=5,
       label='Min f2 (x=3)')

ax.set_xlabel('$f_1(x) = (x-1)^2$')
ax.set_ylabel('$f_2(x) = (x-3)^2$')
ax.set_title('Objective Space and Pareto Front')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================================
# CLOUD SERVER LOAD OPTIMIZATION LAB
# ============================================================================
# f1(x) = x²              (Power consumption in Watts)
# f2(x) = 1/(1.1 - x)     (Latency in milliseconds)
# ============================================================================

# Define objective functions
def f1(x):
    """Power consumption: f1(x) = x²"""
    return x**2

def f2(x):
    """Latency: f2(x) = 1/(1.1 - x)"""
    return 1 / (1.1 - x)

# Constraint bounds
x_min, x_max = 0.1, 0.9

# Compute normalization values (ideal and anti-ideal points)
# Ideal: minimize each function separately
# f1_ideal: minimize x² on [0.1, 0.9] -> x=0.1
# f2_ideal: minimize 1/(1.1-x) on [0.1, 0.9] -> x=0.1

f1_ideal = f1(x_min)  # = 0.01
f2_ideal = f2(x_min)  # = 1/(1.1-0.1) = 10

# Anti-ideal: maximize each function
# f1_antiideal: maximize x² on [0.1, 0.9] -> x=0.9
# f2_antiideal: maximize 1/(1.1-x) on [0.1, 0.9] -> x=0.9
f1_antiideal = f1(x_max)  # = 0.81
f2_antiideal = f2(x_max)  # = 1/(1.1-0.9) = 5

print("=" * 80)
print("CLOUD SERVER LOAD OPTIMIZATION LAB")
print("=" * 80)
print(f"\nNormalization Values:")
print(f"  f1_ideal = {f1_ideal:.4f}, f1_antiideal = {f1_antiideal:.4f}")
print(f"  f2_ideal = {f2_ideal:.4f}, f2_antiideal = {f2_antiideal:.4f}")

# Normalized objective functions
def norm_f1(x):
    """Normalized f1: [f1(x) - f1_ideal] / [f1_antiideal - f1_ideal]"""
    return (f1(x) - f1_ideal) / (f1_antiideal - f1_ideal)

def norm_f2(x):
    """Normalized f2: [f2(x) - f2_ideal] / [f2_antiideal - f2_ideal]"""
    return (f2(x) - f2_ideal) / (f2_antiideal - f2_ideal)

# Weighted sum scalarized function
def objective(x, w):
    """J(x, w) = w * norm_f1(x) + (1-w) * norm_f2(x)"""
    return w * norm_f1(x) + (1 - w) * norm_f2(x)

# ============================================================================
# TASK 1: Generate at least 40 points on the Pareto Front
# ============================================================================
print("\n" + "=" * 80)
print("TASK 1: GENERATING PARETO FRONT (40+ points)")
print("=" * 80)

n_points = 50  # Generate 50 points for better coverage
weights = np.linspace(0, 1, n_points)
results_list = {
    'w': [],
    'x': [],
    'f1': [],
    'f2': [],
    'norm_f1': [],
    'norm_f2': []
}

# Optimization for each weight
for w in weights:
    res = minimize(
        objective,
        x0=0.5,
        args=(w,),
        bounds=[(x_min, x_max)],
        method='L-BFGS-B'
    )
    
    if res.success:
        x_opt = res.x[0]
        results_list['w'].append(w)
        results_list['x'].append(x_opt)
        results_list['f1'].append(f1(x_opt))
        results_list['f2'].append(f2(x_opt))
        results_list['norm_f1'].append(norm_f1(x_opt))
        results_list['norm_f2'].append(norm_f2(x_opt))

# Convert to numpy arrays
results = {key: np.array(values) for key, values in results_list.items()}

print(f"\nGenerated {len(results['w'])} Pareto-optimal points")
print(f"\nPareto Front Points (first 10):")
print(f"{'w':<8} {'x':<10} {'f1(x)':<12} {'f2(x)':<12} {'norm_f1':<12} {'norm_f2':<12}")
print("-" * 80)
for i in range(min(10, len(results['w']))):
    print(f"{results['w'][i]:<8.3f} {results['x'][i]:<10.4f} {results['f1'][i]:<12.6f} {results['f2'][i]:<12.6f} {results['norm_f1'][i]:<12.6f} {results['norm_f2'][i]:<12.6f}")

print(f"\n{'...':<8}")

print(f"\nPareto Front Points (last 10):")
for i in range(max(0, len(results['w'])-10), len(results['w'])):
    print(f"{results['w'][i]:<8.3f} {results['x'][i]:<10.4f} {results['f1'][i]:<12.6f} {results['f2'][i]:<12.6f} {results['norm_f1'][i]:<12.6f} {results['norm_f2'][i]:<12.6f}")

# ============================================================================
# TASK 2: Visualization
# ============================================================================
print("\n" + "=" * 80)
print("TASK 2: VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Pareto Front (Objective Space)
axes[0, 0].plot(results['f1'], results['f2'], 'r-o', linewidth=2, markersize=6, label='Pareto Front')
axes[0, 0].scatter([results['f1'][0]], [results['f2'][0]], color='green', s=150, marker='s', label='w=0 (f2 priority)', zorder=5)
axes[0, 0].scatter([results['f1'][-1]], [results['f2'][-1]], color='blue', s=150, marker='^', label='w=1 (f1 priority)', zorder=5)
axes[0, 0].set_xlabel('$f_1(x) = x^2$ (Power in Watts)', fontsize=11)
axes[0, 0].set_ylabel('$f_2(x) = 1/(1.1-x)$ (Latency in ms)', fontsize=11)
axes[0, 0].set_title('Pareto Front (Objective Space)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)

# Plot 2: Pareto Set (x vs weight w)
axes[0, 1].plot(results['w'], results['x'], 'b-o', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Weight w (priority to energy savings)', fontsize=11)
axes[0, 1].set_ylabel('Optimal Load Factor x', fontsize=11)
axes[0, 1].set_title('Pareto Set: Optimal x vs Weight w', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([x_min - 0.05, x_max + 0.05])

# Plot 3: Objectives vs Weight w
axes[1, 0].plot(results['w'], results['f1'], 'g-o', linewidth=2, markersize=5, label='$f_1(x)$ - Power')
axes[1, 0].plot(results['w'], results['f2'], 'r-s', linewidth=2, markersize=5, label='$f_2(x)$ - Latency')
axes[1, 0].set_xlabel('Weight w', fontsize=11)
axes[1, 0].set_ylabel('Objective Values', fontsize=11)
axes[1, 0].set_title('Objectives vs Weight w', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

# Plot 4: Trade-off curve with normalized objectives
axes[1, 1].plot(results['w'], results['norm_f1'], 'g-o', linewidth=2, markersize=5, label='Normalized $f_1$')
axes[1, 1].plot(results['w'], results['norm_f2'], 'r-s', linewidth=2, markersize=5, label='Normalized $f_2$')
axes[1, 1].set_xlabel('Weight w', fontsize=11)
axes[1, 1].set_ylabel('Normalized Objective Values', fontsize=11)
axes[1, 1].set_title('Normalized Trade-off', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('pareto_front_cloud_server.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to 'pareto_front_cloud_server.png'")
plt.show()

# ============================================================================
# TASK 3: SLA Analysis - Find weights where latency < 4ms
# ============================================================================
print("\n" + "=" * 80)
print("TASK 3: SLA ANALYSIS (Latency < 4 ms constraint)")
print("=" * 80)

# Find indices where f2 < 4
sla_satisfied = results['f2'] < 4.0
sla_indices = np.where(sla_satisfied)[0]

if len(sla_indices) > 0:
    w_min_sla = results['w'][sla_indices[0]]
    w_max_sla = results['w'][sla_indices[-1]]
    
    print(f"\nSLA Constraint: Latency (f2) < 4 ms")
    print(f"SLA is satisfied for {len(sla_indices)} points")
    print(f"Weight range: w ∈ [{w_min_sla:.3f}, {w_max_sla:.3f}]")
    
    print(f"\nDetailed SLA Analysis:")
    print(f"{'w':<8} {'x':<10} {'f1(x)':<12} {'f2(x)':<12} {'SLA':<10}")
    print("-" * 60)
    
    # Show transitions and key points
    for i in range(len(results['w'])):
        status = "✓ OK" if results['f2'][i] < 4.0 else "✗ FAIL"
        if i == 0 or i == len(results['w'])-1 or sla_satisfied[i] != sla_satisfied[i-1]:
            print(f"{results['w'][i]:<8.3f} {results['x'][i]:<10.4f} {results['f1'][i]:<12.6f} {results['f2'][i]:<12.6f} {status:<10}")
    
    print(f"\nKey Points:")
    # Find critical points
    idx_satisfy = sla_indices[0]
    idx_violate = sla_indices[-1]
    
    if idx_satisfy > 0:
        print(f"  Critical w (transition point): ≈ {results['w'][idx_satisfy]:.4f}")
        print(f"    Just before: f2({results['x'][idx_satisfy-1]:.4f}) = {results['f2'][idx_satisfy-1]:.4f} ms")
        print(f"    Just after:  f2({results['x'][idx_satisfy]:.4f}) = {results['f2'][idx_satisfy]:.4f} ms")
    
    # Find the weight where f2 exactly equals 4
    if idx_satisfy > 0 and idx_satisfy < len(results['w']):
        w_critical = np.interp(4.0, results['f2'][::-1], results['w'][::-1])
        print(f"\n  Precise critical weight where f2(x) = 4.0 ms: w ≈ {w_critical:.4f}")
else:
    print("\nNo points satisfy the SLA constraint (f2 < 4 everywhere)")

# Create SLA visualization
fig, ax = plt.subplots(figsize=(12, 6))

# Plot latency curve
ax.plot(results['w'], results['f2'], 'b-o', linewidth=2.5, markersize=6, label='Latency f₂(x)')
ax.axhline(y=4.0, color='r', linestyle='--', linewidth=2, label='SLA Limit (4 ms)')

# Fill regions
ax.fill_between(results['w'], 0, results['f2'], where=(results['f2'] < 4.0).tolist(), 
                alpha=0.3, color='green', label='SLA Satisfied')
ax.fill_between(results['w'], 0, results['f2'], where=(results['f2'] >= 4.0).tolist(), 
                alpha=0.3, color='red', label='SLA Violated')

ax.set_xlabel('Weight w (priority to energy savings)', fontsize=12)
ax.set_ylabel('Latency f₂(x) [milliseconds]', fontsize=12)
ax.set_title('SLA Analysis: Latency Constraint (f₂ < 4 ms)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim((float(np.min(results['f2']) - 0.5), float(np.max(results['f2']) + 0.5)))

# Add annotation for SLA region
if len(sla_indices) > 0:
    w_mid = float((results['w'][sla_indices[0]] + results['w'][sla_indices[-1]]) / 2)
    ax.text(w_mid, 4.5, f'SLA Region\nw ∈ [{w_min_sla:.3f}, {w_max_sla:.3f}]', 
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('sla_analysis.png', dpi=150, bbox_inches='tight')
print("\nSLA analysis plot saved to 'sla_analysis.png'")
plt.show()

print("\n" + "=" * 80)
print("LAB COMPLETE")
print("=" * 80)

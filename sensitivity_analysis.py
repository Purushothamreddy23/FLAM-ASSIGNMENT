import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Load the data
data = pd.read_csv('xy_data.csv')
x_data = data['x'].values
y_data = data['y'].values

# equations
def parametric_curve(t, theta, M, X):
    exp_term = np.exp(M * np.abs(t))
    sin_03t = np.sin(0.3 * t)
    
    x = t * np.cos(theta) - exp_term * sin_03t * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + exp_term * sin_03t * np.cos(theta)
    
    return x, y

def generate_curve(theta, M, X, t_values):
    x_curve, y_curve = parametric_curve(t_values, theta, M, X)
    return x_curve, y_curve

def compute_l1_distance(theta, M, X):
    t_samples = np.linspace(6, 60, 500)
    x_curve, y_curve = generate_curve(theta, M, X, t_samples)
    
    total_distance = 0
    for i in range(len(x_data)):
        distances = np.sqrt((x_curve - x_data[i])**2 + (y_curve - y_data[i])**2)
        min_distance = np.min(distances)
        total_distance += min_distance
    
    return total_distance

# Optimal values from previous optimization
theta_opt = 0.523613
M_opt = 0.030000
X_opt = 55.0007

print("=" * 70)
print("SENSITIVITY ANALYSIS")
print("=" * 70)


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Sensitivity to theta
print("\n1. Testing sensitivity to θ (theta)...")
theta_range = np.linspace(0, np.deg2rad(50), 50)
theta_errors = [compute_l1_distance(th, M_opt, X_opt) for th in theta_range]

axes[0, 0].plot(np.rad2deg(theta_range), theta_errors, 'b-', linewidth=2)
axes[0, 0].axvline(np.rad2deg(theta_opt), color='r', linestyle='--', label=f'Optimal: {np.rad2deg(theta_opt):.2f}°')
axes[0, 0].set_xlabel('θ (degrees)')
axes[0, 0].set_ylabel('L1 Distance')
axes[0, 0].set_title('Sensitivity to θ')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Sensitivity to M
print("2. Testing sensitivity to M...")
M_range = np.linspace(-0.05, 0.05, 50)
M_errors = [compute_l1_distance(theta_opt, m, X_opt) for m in M_range]

axes[0, 1].plot(M_range, M_errors, 'g-', linewidth=2)
axes[0, 1].axvline(M_opt, color='r', linestyle='--', label=f'Optimal: {M_opt:.6f}')
axes[0, 1].set_xlabel('M')
axes[0, 1].set_ylabel('L1 Distance')
axes[0, 1].set_title('Sensitivity to M')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Sensitivity to X
print("3. Testing sensitivity to X...")
X_range = np.linspace(0, 100, 50)
X_errors = [compute_l1_distance(theta_opt, M_opt, x) for x in X_range]

axes[0, 2].plot(X_range, X_errors, 'm-', linewidth=2)
axes[0, 2].axvline(X_opt, color='r', linestyle='--', label=f'Optimal: {X_opt:.2f}')
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('L1 Distance')
axes[0, 2].set_title('Sensitivity to X')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. 2D contour: theta vs M
print("4. Creating θ vs M contour plot...")
theta_2d = np.linspace(np.deg2rad(20), np.deg2rad(40), 30)
M_2d = np.linspace(0.01, 0.05, 30)
errors_2d = np.zeros((len(M_2d), len(theta_2d)))

for i, m in enumerate(M_2d):
    for j, th in enumerate(theta_2d):
        errors_2d[i, j] = compute_l1_distance(th, m, X_opt)

contour = axes[1, 0].contourf(np.rad2deg(theta_2d), M_2d, errors_2d, levels=20, cmap='viridis')
axes[1, 0].plot(np.rad2deg(theta_opt), M_opt, 'r*', markersize=15, label='Optimal')
axes[1, 0].set_xlabel('θ (degrees)')
axes[1, 0].set_ylabel('M')
axes[1, 0].set_title('θ vs M (X fixed)')
axes[1, 0].legend()
plt.colorbar(contour, ax=axes[1, 0], label='L1 Distance')

# 5. 2D contour: theta vs X
print("5. Creating θ vs X contour plot...")
X_2d = np.linspace(40, 70, 30)
errors_2d_X = np.zeros((len(X_2d), len(theta_2d)))

for i, x in enumerate(X_2d):
    for j, th in enumerate(theta_2d):
        errors_2d_X[i, j] = compute_l1_distance(th, M_opt, x)

contour2 = axes[1, 1].contourf(np.rad2deg(theta_2d), X_2d, errors_2d_X, levels=20, cmap='plasma')
axes[1, 1].plot(np.rad2deg(theta_opt), X_opt, 'r*', markersize=15, label='Optimal')
axes[1, 1].set_xlabel('θ (degrees)')
axes[1, 1].set_ylabel('X')
axes[1, 1].set_title('θ vs X (M fixed)')
axes[1, 1].legend()
plt.colorbar(contour2, ax=axes[1, 1], label='L1 Distance')

# 6. 2D contour: M vs X
print("6. Creating M vs X contour plot...")
errors_2d_MX = np.zeros((len(X_2d), len(M_2d)))

for i, x in enumerate(X_2d):
    for j, m in enumerate(M_2d):
        errors_2d_MX[i, j] = compute_l1_distance(theta_opt, m, x)

contour3 = axes[1, 2].contourf(M_2d, X_2d, errors_2d_MX, levels=20, cmap='coolwarm')
axes[1, 2].plot(M_opt, X_opt, 'r*', markersize=15, label='Optimal')
axes[1, 2].set_xlabel('M')
axes[1, 2].set_ylabel('X')
axes[1, 2].set_title('M vs X (θ fixed)')
axes[1, 2].legend()
plt.colorbar(contour3, ax=axes[1, 2], label='L1 Distance')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
print("\nSensitivity analysis saved!")

# Print summary statistics
print("\n" + "=" * 70)
print("PARAMETER SENSITIVITY SUMMARY")
print("=" * 70)
print(f"\nOptimal parameters:")
print(f"  θ = {np.rad2deg(theta_opt):.4f}° ({theta_opt:.6f} rad)")
print(f"  M = {M_opt:.6f}")
print(f"  X = {X_opt:.4f}")
print(f"\nOptimal L1 distance: {compute_l1_distance(theta_opt, M_opt, X_opt):.4f}")

# Test parameter variations
variations = [0.01, 0.05, 0.1]
print("\nImpact of parameter variations:")
for var in variations:
    theta_var = theta_opt * (1 + var)
    M_var = M_opt * (1 + var)
    X_var = X_opt * (1 + var)
    
    err_theta = compute_l1_distance(theta_var, M_opt, X_opt)
    err_M = compute_l1_distance(theta_opt, M_var, X_opt)
    err_X = compute_l1_distance(theta_opt, M_opt, X_var)
    
    print(f"\n{var*100:.0f}% increase:")
    print(f"  θ variation: L1 = {err_theta:.4f} (Δ = {err_theta - compute_l1_distance(theta_opt, M_opt, X_opt):.4f})")
    print(f"  M variation: L1 = {err_M:.4f} (Δ = {err_M - compute_l1_distance(theta_opt, M_opt, X_opt):.4f})")
    print(f"  X variation: L1 = {err_X:.4f} (Δ = {err_X - compute_l1_distance(theta_opt, M_opt, X_opt):.4f})")

print("\n" + "=" * 70)

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('xy_data.csv')
x_data = data['x'].values
y_data = data['y'].values

print(f"Data shape: {data.shape}")
print(f"X range: [{x_data.min():.2f}, {x_data.max():.2f}]")
print(f"Y range: [{y_data.min():.2f}, {y_data.max():.2f}]")
print(f"Number of points: {len(x_data)}")

# Parametric curve equations
def parametric_curve(t, theta, M, X):
    """
    x(t) = t*cos(theta) - e^(M|t|)*sin(0.3t)*sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M|t|)*sin(0.3t)*cos(theta)
    """
    exp_term = np.exp(M * np.abs(t))
    sin_03t = np.sin(0.3 * t)
    
    x = t * np.cos(theta) - exp_term * sin_03t * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + exp_term * sin_03t * np.cos(theta)
    
    return x, y

# Generate curve for given parameters
def generate_curve(theta, M, X, t_values):
    x_curve, y_curve = parametric_curve(t_values, theta, M, X)
    return x_curve, y_curve

# Objective function: minimize distance between data points and curve
def objective_function(params):
   
    theta, M, X = params
    
    # Generate dense sampling of the curve
    t_samples = np.linspace(6, 60, 500)
    x_curve, y_curve = generate_curve(theta, M, X, t_samples)
    
    # For each data point, find minimum distance to curve
    total_distance = 0
    for i in range(len(x_data)):
        # Calculate distance from this data point to all curve points
        distances = np.sqrt((x_curve - x_data[i])**2 + (y_curve - y_data[i])**2)
        min_distance = np.min(distances)
        total_distance += min_distance
    
    return total_distance

# Alternative objective: try to match with estimated t values
def objective_with_t_estimation(params):
    
    theta, M, X = params
    
    total_error = 0
    
    for i in range(len(x_data)):
        x_obs = x_data[i]
        y_obs = y_data[i]
        
        # Try different t values and find the one that gives closest point
        t_test = np.linspace(6, 60, 200)
        x_pred, y_pred = generate_curve(theta, M, X, t_test)
        
        # to know the closest point
        distances = np.sqrt((x_pred - x_obs)**2 + (y_pred - y_obs)**2)
        min_dist = np.min(distances)
        total_error += min_dist
    
    return total_error

print("\nStarting optimization...")
print("=" * 60)


bounds = [
    (0, np.deg2rad(50)),  # theta in radians
    (-0.05, 0.05),         # M
    (0, 100)               # X
]

# Use differential evolution for global optimization
print("\nMethod 1: Differential Evolution (Global Search)")
result_de = differential_evolution(
    objective_function,
    bounds=bounds,
    seed=42,
    maxiter=300,
    popsize=20,
    atol=1e-6,
    tol=1e-6,
    workers=1,
    updating='deferred',
    polish=True
)

theta_opt, M_opt, X_opt = result_de.x
theta_deg = np.rad2deg(theta_opt)

print(f"\nOptimization Result:")
print(f"  θ (theta) = {theta_opt:.6f} radians = {theta_deg:.4f} degrees")
print(f"  M = {M_opt:.6f}")
print(f"  X = {X_opt:.4f}")
print(f"  Total L1 distance: {result_de.fun:.4f}")
print(f"  Average distance per point: {result_de.fun/len(x_data):.4f}")

# Generate the fitted curve
t_fitted = np.linspace(6, 60, 1000)
x_fitted, y_fitted = generate_curve(theta_opt, M_opt, X_opt, t_fitted)

# Create visualization
plt.figure(figsize=(12, 8))

# Plot data points
plt.subplot(2, 2, 1)
plt.scatter(x_data, y_data, c='blue', s=20, alpha=0.6, label='Data points')
plt.plot(x_fitted, y_fitted, 'r-', linewidth=2, label='Fitted curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Parametric Curve Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Plot x vs t (estimated)
plt.subplot(2, 2, 2)
# Estimate t for each data point
t_estimates = []
for i in range(len(x_data)):
    t_test = np.linspace(6, 60, 500)
    x_pred, y_pred = generate_curve(theta_opt, M_opt, X_opt, t_test)
    distances = np.sqrt((x_pred - x_data[i])**2 + (y_pred - y_data[i])**2)
    best_idx = np.argmin(distances)
    t_estimates.append(t_test[best_idx])

t_estimates = np.array(t_estimates)
plt.scatter(t_estimates, x_data, c='blue', s=20, alpha=0.6, label='Data')
plt.plot(t_fitted, x_fitted, 'r-', linewidth=2, label='Fitted')
plt.xlabel('t')
plt.ylabel('X')
plt.title('X component vs t')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot y vs t
plt.subplot(2, 2, 3)
plt.scatter(t_estimates, y_data, c='blue', s=20, alpha=0.6, label='Data')
plt.plot(t_fitted, y_fitted, 'r-', linewidth=2, label='Fitted')
plt.xlabel('t')
plt.ylabel('Y')
plt.title('Y component vs t')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot residuals
plt.subplot(2, 2, 4)
residuals = []
for i in range(len(x_data)):
    t_test = np.linspace(6, 60, 500)
    x_pred, y_pred = generate_curve(theta_opt, M_opt, X_opt, t_test)
    distances = np.sqrt((x_pred - x_data[i])**2 + (y_pred - y_data[i])**2)
    residuals.append(np.min(distances))

plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Distance to curve')
plt.ylabel('Frequency')
plt.title(f'Residual Distribution (mean: {np.mean(residuals):.4f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/curve_fitting_results.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: curve_fitting_results.png")

# Calculate L1 distance metric
print("\n" + "=" * 60)
print("ACCURACY METRICS:")
print("=" * 60)
print(f"Total L1 distance: {result_de.fun:.4f}")
print(f"Average distance per point: {np.mean(residuals):.4f}")
print(f"Max distance: {np.max(residuals):.4f}")
print(f"Min distance: {np.min(residuals):.4f}")
print(f"Std deviation: {np.std(residuals):.4f}")


print("\n" + "=" * 60)
print("SUBMISSION FORMAT (LaTeX):")
print("=" * 60)
latex_submission = (
    f"\\left(t*\\cos({theta_opt:.6f})-e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_opt:.6f})\\ "
    f"+{X_opt:.4f},42+\\ t*\\sin({theta_opt:.6f})+e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_opt:.6f})\\right)"
)
print(latex_submission)


print("\n" + "=" * 60)
print("SUBMISSION FORMAT (with theta in degrees for reference):")
print("=" * 60)
print(f"θ = {theta_deg:.4f}°")
print(f"M = {M_opt:.6f}")
print(f"X = {X_opt:.4f}")

print("\nDesmos format (copy to https://www.desmos.com/calculator):")
desmos_format = (
    f"\\left(t*\\cos({theta_opt:.4f})-e^{{{M_opt:.4f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_opt:.4f})\\ "
    f"+{X_opt:.2f},42+\\ t*\\sin({theta_opt:.4f})+e^{{{M_opt:.4f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_opt:.4f})\\right)"
)
print(desmos_format)

plt.show()

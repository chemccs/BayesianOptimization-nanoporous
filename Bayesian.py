import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import seaborn as sns

# Objective function: Smooth and continuous
def objective_function(x):
    return np.sin(x) * np.cos(2 * x)

# Generate data
X = np.linspace(-3, 3, 400).reshape(-1, 1)
Y = objective_function(X)

# Observations
X_sample = np.array([-2.5, -1.5, 0, 1.5, 2.]).reshape(-1, 1)
Y_sample = objective_function(X_sample)

# Define the kernel with initial hyperparameters
kernel = Matern(nu=2.5, length_scale=1.)

# Initialize Gaussian process regressor with the ability to optimize hyperparameters
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                              n_restarts_optimizer=10)

gp.fit(X_sample, Y_sample)
Y_pred, sigma = gp.predict(X, return_std=True)

# Correct the shape of sigma to match with Y_pred
sigma = sigma.reshape(-1, 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X, Y, 'r:', label='True Function')
plt.errorbar(X.ravel(), Y_pred.ravel(), yerr=1.96 * sigma.ravel(), label='GP Prediction ± 1.96*σ', alpha=.5)
plt.scatter(X_sample, Y_sample, s=40, zorder=3, color='black', label='Observations')
plt.xlabel('MOF feature', fontsize=14, weight='bold')
plt.ylabel('Target value', fontsize=14, weight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlim([-3, 3])
plt.legend()
plt.grid(True)
plt.savefig("Baysian.png", dpi=500)


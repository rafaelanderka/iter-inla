import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve
from findiff import FinDiff, Coef, Identity

from spdeinf import nonlinear, util

# Parameters
alpha = 0.1

# Domain
N = 40  # Grid size
x = y = np.linspace(-2, 2, N)
X, Y  = np.meshgrid(x, y, indexing='ij')
dx = x[1] - x[0]
dy = y[1] - y[0]

# Define the function f(x)
def f(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

# Define the differential operators
laplacian = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)

def equation(u):
    u = u.reshape((N, N))
    
    result = alpha * np.cos(u) - laplacian(u) - f(X, Y)
    
    # Propagate boundary conditions
    result[0, :] = u[0, :]
    result[-1, :] = u[-1, :]
    result[:, 0] = u[:, 0]
    result[:, -1] = u[:, -1]
    
    return result.ravel()

# Initial guess
u0 = np.zeros((N, N))

# Solve the system
u = fsolve(equation, u0.ravel()).reshape((N, N))

# Plot ground truth solution
plt.imshow(u, origin="lower")
plt.colorbar()
plt.show()

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dx, dy, alpha):
    """
    Constructs current linearised differential operator.
    """
    partial_xx = FinDiff(0, dx, 2)
    partial_yy = FinDiff(1, dy, 2)
    sin_u0 = np.sin(u0)
    diff_op = Coef(-alpha * sin_u0) * Identity() - partial_xx - partial_yy
    return diff_op

def get_prior_mean(u0, alpha, diff_op_gen):
    """
    Calculates current prior mean.
    """
    sin_u0 = np.sin(u0)
    cos_u0 = np.cos(u0)
    diff_op = diff_op_gen(u0)
    diff_op_mat = diff_op.matrix(u0.shape)
    rhs = f(X, Y) - alpha * (cos_u0 + sin_u0 * u0)
    prior_mean = spsolve(diff_op_mat, rhs.flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u: get_diff_op(u, dx, dy, alpha)
prior_mean_gen = lambda u: get_prior_mean(u, alpha, diff_op_gen)

######################################
# Naive diff. op. and mean generator #
######################################

def get_diff_op_naive(u0, dx, dy, alpha):
    """
    Constructs naively linearised differential operator.
    """
    partial_xx = FinDiff(0, dx, 2)
    partial_yy = FinDiff(1, dy, 2)
    diff_op = (-1) * partial_xx - partial_yy
    return diff_op

def get_prior_mean_naive(u0, diff_op_gen):
    diff_op = diff_op_gen(u0)
    diff_op_mat = diff_op.matrix(u0.shape)
    cos_u0 = np.cos(u0)
    rhs = f(X, Y) - alpha * cos_u0 
    prior_mean = spsolve(diff_op_mat, rhs.flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen_naive = lambda u: get_diff_op_naive(u, dx, dy, alpha)
prior_mean_gen_naive = lambda u: get_prior_mean_naive(u, diff_op_gen)

## Fit GP with non-linear SPDE prior from elliptic equation

# Sample observations
obs_std = 1e-4
obs_count = 20
obs_dict = util.sample_observations(u, obs_count, obs_std)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)

# Fit with expansion linearisation
max_iter = 20
model = nonlinear.NonlinearSPDERegressor(u, dx, dy, diff_op_gen, prior_mean_gen)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True)
iter_count = len(model.mse_hist)

# Fit with naive linearisation
model_naive = nonlinear.NonlinearSPDERegressor(u, dx, dy, diff_op_gen_naive, prior_mean_gen_naive)
model_naive.fit(obs_dict, obs_std, max_iter=max_iter, animated=True)
iter_count_naive = len(model_naive.mse_hist)

# Plot convergence history
plt.plot(np.arange(1, iter_count + 1), model.mse_hist, label="Linearisation via expansion")
plt.plot(np.arange(1, iter_count_naive + 1), model_naive.mse_hist, label="Naive linearisation")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.xticks(np.arange(2, max(iter_count, iter_count_naive) + 1, 2))
plt.legend()
plt.savefig("figures/elliptic/mse_conv.png", dpi=200)
plt.show()

# Save animation
print("Saving animation...")
model.save_animation("figures/elliptic/elliptic_iter_animation.gif", fps=2)

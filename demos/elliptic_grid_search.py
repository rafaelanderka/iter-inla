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

# Plot solution
plt.imshow(u, origin="lower")
plt.colorbar()
plt.show()

# Sample observations
obs_noise = 1e-4
obs_count = 20
obs_dict = util.sample_observations(u, obs_count, obs_noise)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)

# Perform grid search of best alpha based on MSE
N_alphas = 40
true_alpha = alpha
alphas = np.linspace(0.05, 0.15, N_alphas)
mses = np.zeros(N_alphas)
for i, alpha in enumerate(alphas):
    print("Evaluating alpha =", alpha)

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

    def get_diff_op_naive(u0, dx, dy, alpha):
        """
        Constructs naively linearised differential operator.
        """
        partial_xx = FinDiff(0, dx, 2)
        partial_yy = FinDiff(1, dy, 2)
        diff_op = (-1)* partial_xx - partial_yy
        return diff_op

    diff_op_gen = lambda u: get_diff_op(u, dx, dy, alpha)
    prior_mean_gen = lambda u: get_prior_mean(u, alpha, diff_op_gen)


    ######################################
    # Naive diff. op. and mean generator #
    ######################################

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
    # Perform iterative optimisation
    max_iter = 20
    model = nonlinear.NonlinearSPDERegressor(u, dx, dy, diff_op_gen, prior_mean_gen)
    model.fit(obs_dict, obs_noise, max_iter=max_iter)
    mses[i] = model.mse

# Plot results of grid search
plt.plot(alphas, mses)
plt.xlabel("$\\alpha$")
plt.ylabel("MSE")
plt.axvline(alphas[np.argmin(mses)], color='darkred', ls='--', label="Optimal $\\alpha$")
plt.axvline(true_alpha, color='grey', ls=':', label="True $\\alpha$")
plt.legend()
plt.tight_layout()
plt.savefig("figures/elliptic/mse_grid_search.png", dpi=200)
plt.show()

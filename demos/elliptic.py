import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve
from findiff import FinDiff, Coef, Identity

from spdeinf import util
from spdeinf.nonlinear import SPDEDynamics, IterativeRegressor

# Set seed
np.random.seed(0)

####################################
# Generate data from elliptic eqn. #
####################################

# Define parameters of elliptic equation and model
alpha = 0.1
obs_std = 1e-4
params = np.array([alpha, obs_std])

# Define domain
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

# Define elliptic equation
def elliptic(u):
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
u = fsolve(elliptic, u0.ravel()).reshape((N, N))

# Sample observations
obs_count = 20
obs_dict = util.sample_observations(u, obs_count, obs_std)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)

#################################
# Define elliptic eqn. dynamics #
#################################

class EllipticDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. alpha
    1. observation noise
    """

    def __init__(self, dx, dy) -> None:
        super().__init__()
        self.dx = dx
        self.dy = dy

    def get_diff_op(self, u0, params, **kwargs):
        """
        Construct current linearised differential operator.
        """
        alpha, _ = params
        partial_xx = FinDiff(0, dx, 2)
        partial_yy = FinDiff(1, dy, 2)
        sin_u0 = np.sin(u0)
        diff_op = Coef(-alpha * sin_u0) * Identity() - partial_xx - partial_yy
        return diff_op

    def get_prior_precision(self, u0, params, **kwargs):
        """
        Calculate current prior precision.
        """
        diff_op_guess = self.get_diff_op(u0, params, **kwargs)
        L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
        # prior_precision = (self.dy * self.dx / params[2]**2) * (L.T @ L)
        # prior_precision = self.dy * self.dx * (L.T @ L)
        prior_precision = L.T @ L
        return prior_precision

    def get_prior_mean(self, u0, params, **kwargs):
        """
        Calculate current prior mean.
        """
        # Construct linearisation remainder term
        sin_u0 = np.sin(u0)
        cos_u0 = np.cos(u0)
        remainder = f(X, Y) - alpha * (cos_u0 + sin_u0 * u0)

        # Construct diff. op.
        diff_op = self.get_diff_op(u0, params, **kwargs)
        L = diff_op.matrix(u0.shape)

        # Compute prior mean
        prior_mean = spsolve(L, remainder.flatten())
        # prior_mean = spsolve(L.T @ L, L.T @ remainder.flatten())
        return prior_mean.reshape(u0.shape)

    def get_obs_noise(self, params, **kwargs):
        """
        Get observation noise (standard deviation).
        """
        return params[1]

dynamics = EllipticDynamics(dx, dy)

##########################################
# Fit model with iterative linearistaion #
##########################################

max_iter = 20
model = IterativeRegressor(u, dynamics, mixing_coef=1.)
model.fit(obs_dict, params, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)

############
# Plot fit #
############

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_mean, extent=(0, 4, -4, 4), origin="lower", aspect=.5, rasterized=True, cmap="RdBu_r")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dy * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/elliptic/elliptic_spde.pdf", transparent=True)

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_std, extent=(0, 4, -4, 4), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dy * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/elliptic/elliptic_spde_std.pdf", transparent=True)
plt.show()

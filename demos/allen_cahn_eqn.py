import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from findiff import FinDiff, Coef, Identity

from spdeinf import util
from spdeinf.nonlinear import SPDEDynamics, IterativeRegressor

# Set seed
np.random.seed(0)

##################################
# Load data from Allen-Cahn eqn. #
##################################

# Define parameters of the Allen-Cahn eqn. and model
alpha = 0.001
beta = 5
obs_std = 1e-2
params = np.array([alpha, beta, obs_std])

# Load Allen-Cahn eqn. data from PINNs examples
data = loadmat("data/PINNs/AC.mat")
uu = data['uu']
xx = data['x'].squeeze()
tt = data['tt'].squeeze()
N_x = xx.shape[0]
N_t = tt.shape[0]
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]

# Define Allen-Cahn eq. parameters
alpha = 0.001
beta = 5

# Sample observations
obs_count = 256
obs_dict = util.sample_observations(uu, obs_count, obs_std, extent=(None, None, 0, 56))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
print("Number of observations:", obs_idxs.shape[0])

###################################
# Define Allen-Cahn eqn. dynamics #
###################################

class ACDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. alpha
    1. beta
    2. observation noise
    """

    def __init__(self, dx, dt) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt

    def get_diff_op(self, u0, params, **kwargs):
        """
        Construct current linearised differential operator.
        """
        alpha, beta, _ = params
        partial_t = FinDiff(1, dt, 1, acc=2)
        partial_xx = FinDiff(0, dx, 2, acc=2, periodic=True)
        u0_sq = u0 ** 2
        diff_op = partial_t - Coef(alpha) * partial_xx + Coef(3 * beta * u0_sq) * Identity() - Coef(beta) * Identity()
        return diff_op

    def get_prior_precision(self, u0, params, **kwargs):
        """
        Calculate current prior precision.
        """
        diff_op_guess = self.get_diff_op(u0, params, **kwargs)
        L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
        # prior_precision = (self.dt * self.dx / params[2]**2) * (L.T @ L)
        # prior_precision = self.dt * self.dx * (L.T @ L)
        prior_precision = L.T @ L
        return prior_precision

    def get_prior_mean(self, u0, params, **kwargs):
        """
        Calculate current prior mean.
        """
        _, beta, _ = params

        # Construct linearisation remainder term
        u0_cu = u0 ** 3
        remainder = 2 * beta * u0_cu

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
        return params[2]

dynamics = ACDynamics(dx, dt)

##########################################
# Fit model with iterative linearistaion #
##########################################

max_iter = 20
model = IterativeRegressor(uu, dynamics, mixing_coef=0.5)
model.fit(obs_dict, params, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)

############
# Plot fit #
############

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_mean, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_spde.pdf", transparent=True)

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_std, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r", norm=LogNorm(vmin=1e-5, vmax=3))
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_spde_std.pdf", transparent=True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from findiff import FinDiff, Coef, Identity

from spdeinf import util
from spdeinf.nonlinear import SPDEDynamics, IterativeRegressor

# Set seed
np.random.seed(0)

#########################################
# Load data from Korteweg-de Vries eqn. #
#########################################

# Define parameters of the Korteweg-de Vries eqn. and model
l1 = 1
l2 = 0.0025
obs_std = 1e-3
params = np.array([l1, l2, obs_std])

# Load Korteweg-de Vries eqn. data from PINNs examples
data = loadmat("data/PINNs/KdV.mat")
uu = data['uu'][::4,::4]
xx = data['x'].squeeze()[::4]
tt = data['tt'].squeeze()[::4]
N_x = xx.shape[0]
N_t = tt.shape[0]
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]
shape = (N_x, N_t)

# Sample observations
obs_count_1 = 20
obs_count_2 = 20
obs_loc_1 = np.where(tt == 0.2)[0][0]
obs_loc_2 = np.where(tt == 0.8)[0][0]
obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_1+1))
obs_dict.update(util.sample_observations(uu, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()), dtype=float)
obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
print("Number of observations:", obs_idxs.shape[0])

##########################################
# Define Korteweg-de Vries eqn. dynamics #
##########################################

class KdVDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. l1
    1. l2
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
        l1, l2, _ = params
        partial_t = FinDiff(1, self.dt, 1, acc=2)
        partial_x = FinDiff(0, self.dx, 1, acc=2, periodic=True)
        partial_xxx = FinDiff(0, self.dx, 3, acc=2, periodic=True)
        u0_x = partial_x(u0)
        diff_op = partial_t + Coef(l1 * u0) * partial_x + Coef(l1 * u0_x) * Identity() + Coef(l2) * partial_xxx
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
        l1, _, _ = params

        # Construct linearisation remainder term
        partial_x = FinDiff(0, self.dx, 1, acc=2, periodic=True)
        u0_x = partial_x(u0)
        remainder = l1 * u0 * u0_x

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

dynamics = KdVDynamics(dx, dt)

##########################################
# Fit model with iterative linearistaion #
##########################################

max_iter = 20
model = IterativeRegressor(uu, dynamics, mixing_coef=0.1)
model.fit(obs_dict, params, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)

############
# Plot fit #
############

plt.figure(figsize=(3,3))
im_mean = plt.imshow(model.posterior_mean, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="gnuplot2")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="white", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.colorbar(im_mean)
plt.tight_layout(pad=0)
plt.savefig("figures/kdv/kdv_spde.pdf", transparent=True)

plt.figure(figsize=(3,3))
im_std = plt.imshow(model.posterior_std, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.colorbar(im_std)
plt.tight_layout(pad=0)
plt.savefig("figures/kdv/kdv_spde_std.pdf", transparent=True)
plt.show()

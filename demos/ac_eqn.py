import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from findiff import FinDiff, Coef, Identity

from iinla import util
from iinla.nonlinear import SPDEDynamics, IterativeRegressor

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
uu_full = data['uu']
xx_full = data['x'].squeeze()
tt_full = data['tt'].squeeze()
dx_full = xx_full[1] - xx_full[0]
dt_full = tt_full[1] - tt_full[0]
uu = uu_full[::4,::4]
xx = xx_full[::4]
print(xx[0], xx[-1])
tt = tt_full[::4]
print(tt[0], tt[-1])
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]
N_x = xx.shape[0]
N_t = tt.shape[0]
shape = (N_x, N_t)
Xgrid, Tgrid = np.meshgrid(xx, tt, indexing='ij')
print(dx, dt)
print(dx_full, dt_full)


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

    def _update_diff_op(self):
        """
        Construct current linearised differential operator.
        """
        alpha = self._params[0]
        beta = self._params[1]
        partial_t = FinDiff(1, self.dt, 1, acc=2)
        partial_xx = FinDiff(0, self.dx, 2, acc=2, periodic=True)
        u0_sq = self._u0 ** 2
        diff_op = partial_t - Coef(alpha) * partial_xx + Coef(3 * beta * u0_sq) * Identity() - Coef(beta) * Identity()
        return diff_op

    def _update_prior_precision(self):
        """
        Calculate current prior precision.
        """
        # prior_precision = (self.dt * self.dx / self._params[2]**2) * (self._L.T @ self._L)
        # prior_precision = self.dt * self.dx * (self._L.T @ self._L)
        prior_precision = self._L.T @ self._L
        return prior_precision

    def _update_prior_mean(self):
        """
        Calculate current prior mean.
        """
        beta = self._params[1]

        # Construct linearisation remainder term
        u0_cu = self._u0 ** 3
        remainder = 2 * beta * u0_cu

        # Compute prior mean
        prior_mean = spsolve(self._L, remainder.flatten())
        # prior_mean = spsolve(self._L.T @ self._L, self._L.T @ remainder.flatten())
        return prior_mean.reshape(self._u0.shape)

    def _update_obs_noise(self):
        """
        Get observation noise (standard deviation).
        """
        return self._params[2]

dynamics = ACDynamics(dx, dt)

##########################################
# Fit model with iterative linearisation #
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

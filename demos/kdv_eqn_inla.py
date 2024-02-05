import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
from findiff import FinDiff, Coef, Identity

from spdeinf import util
from spdeinf.nonlinear import SPDEDynamics, IterativeINLARegressor
from spdeinf.distributions import LogNormal

np.random.seed(2)

#########################################
# Load data from Korteweg-de Vries eqn. #
#########################################

# Define parameters of the Korteweg-de Vries eqn.
l1 = 1
l2 = 0.0025
obs_std = 1e-3
params_true = np.array([l1, l2, obs_std])
print("True parameters:", params_true)

# Define parameters of the model parameter priors
tau_l1 = 0.1
l1_prior_mode = 1.0
l1_0 = np.log(l1_prior_mode) + (tau_l1 ** (-2))

tau_l2 = 0.1
l2_prior_mode = 0.0025
l2_0 = np.log(l2_prior_mode) + (tau_l2 ** (-2))

# Process noise prior
tau_k = 2
k_prior_mode = 0.05
k_0 = np.log(k_prior_mode) + (tau_k ** (-2))

# Observation noise prior
tau_s = 0.1
s_prior_mode = 0.01
s_0 = np.log(s_prior_mode) + (tau_s ** (-2))

# param0 = np.array([l1_prior_mode, l2_prior_mode, k_prior_mode, s_prior_mode])
# param_priors = [LogNormal(mu=l1_0, sigma=1/tau_l1), LogNormal(mu=l2_0, sigma=1/tau_l2),
#                 LogNormal(mu=k_0, sigma=1/tau_k), LogNormal(mu=s_0, sigma=1/tau_s)]
# param_bounds = [(0.5, 1.5), (0.001, 0.005), (0, 1), (0, 0.1)]

param0 = np.array([l1_prior_mode, l2_prior_mode, s_prior_mode])
param_priors = [LogNormal(mu=l1_0, sigma=1/tau_l1), LogNormal(mu=l2_0, sigma=1/tau_l2),
                LogNormal(mu=s_0, sigma=1/tau_s)]
param_bounds = [(0.5, 1.5), (0.001, 0.005), (0, 0.1)]

# Load Korteweg-de Vries eq. data from PINNs examples
data = loadmat("data/PINNs/KdV.mat")
uu = data['uu'][::4,::4]
xx = data['x'].squeeze()[::4]
tt = data['tt'].squeeze()[::4]
N_x = xx.shape[0]
N_t = tt.shape[0]
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]

# Sample observations
obs_count_1 = 50
obs_count_2 = 50
obs_loc_1 = np.where(tt == 0.2)[0][0]
obs_loc_2 = np.where(tt == 0.8)[0][0] + 1
obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_1+1))
obs_dict.update(util.sample_observations(uu, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
print("Number of observations:", obs_idxs.shape[0])

##########################################
# Define Korteweg-de Vries eqn. dynamics #
##########################################

class KdVDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. l1
    1. l2
    2. process amplitude
    3. observation noise
    """

    def __init__(self, dx, dt) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt

    def get_diff_op(self, u0, params, **kwargs):
        """
        Construct current linearised differential operator.
        """
        # l1, l2, _, _ = params
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
        diff_op_guess = self.get_diff_op(u0, params)
        L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
        # prior_precision = (self.dt * self.dx / params[2]**2) * (L.T @ L)
        prior_precision = self.dt * self.dx * (L.T @ L)
        # prior_precision = L.T @ L
        return prior_precision

    def get_prior_mean(self, u0, params, **kwargs):
        """
        Calculate current prior mean.
        """
        # l1, _, _, _ = params
        l1, _, _ = params

        # Construct linearisation remainder term
        partial_x = FinDiff(0, self.dx, 1, acc=2, periodic=True)
        u0_x = partial_x(u0)
        remainder = l1 * u0 * u0_x

        # Construct diff. op.
        diff_op = self.get_diff_op(u0, params)
        L = diff_op.matrix(u0.shape)

        # Compute prior mean
        # prior_mean = spsolve(L, remainder.flatten())
        prior_mean = spsolve(L.T @ L, L.T @ remainder.flatten())
        return prior_mean.reshape(u0.shape)

    def get_obs_noise(self, params, **kwargs):
        """
        Get observation noise (standard deviation).
        """
        # return params[3]
        return params[2]

dynamics = KdVDynamics(dx, dt)

#################################
# Fit model with iterative INLA #
#################################

max_iter = 20
parameterisation = 'natural' # 'moment' or 'natural'
model = IterativeINLARegressor(uu, dynamics, param0,
                               mixing_coef=0.5,
                               param_bounds=param_bounds,
                               param_priors=param_priors,
                               sampling_evec_scales=[0.1, 0.1, 0.5],
                               sampling_threshold=3)

model.fit(obs_dict, max_iter=max_iter, parameterisation=parameterisation, animated=True, calc_std=False, calc_mnll=True)

############
# Plot fit #
############

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_mean, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="gnuplot2", vmin=-1, vmax=2.5)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="white", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/kdv/kdv_spde_inla.pdf", transparent=True)

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_std, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r", vmin=0, vmax=0.8)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/kdv/kdv_spde_inla_std.pdf", transparent=True)
plt.show()

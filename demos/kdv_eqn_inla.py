import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
from findiff import FinDiff, Coef, Identity

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from iinla import util
from iinla.nonlinear import SPDEDynamics, IterativeINLARegressor
from iinla.distributions import LogNormal

# General configuration
data_id = 3 # 0 - 4
parameterisation = 'natural' # 'moment' or 'natural'
max_iter = 20 # 10 - 20

# Set seed
np.random.seed(0)

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
tau_l1 = 1
l1_prior_mode = 0.5
l1_0 = np.log(l1_prior_mode) + (tau_l1 ** (-2))

# Process noise prior
tau_k = 1
k_prior_mode = 0.01
k_0 = np.log(k_prior_mode) + (tau_k ** (-2))

param0 = np.array([l1_prior_mode, k_prior_mode])
param_priors = [LogNormal(mu=l1_0, sigma=1/tau_l1),
                LogNormal(mu=k_0, sigma=1/tau_k)]
param_bounds = [(0.1, 3.0), (0.001, 0.100)]

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
# obs_count_1 = 20
# obs_count_2 = 20
# obs_loc_1 = np.where(tt == 0.2)[0][0]
# obs_loc_2 = np.where(tt == 0.8)[0][0] + 1
# obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_1+1))
# obs_dict.update(util.sample_observations(uu, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
# obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
# obs_vals = np.array(list(obs_dict.values()), dtype=float)
# obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
# print("Number of observations:", obs_idxs.shape[0])

# Load observations
data_file = f"data/kdv_{data_id}.pkl"
with open(data_file, 'rb') as f:
    data_dict = pickle.load(f)
obs_dict = data_dict['obs_dict']
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
obs_vals = np.array(list(obs_dict.values()), dtype=float)

##########################################
# Define Korteweg-de Vries eqn. dynamics #
##########################################

class KdVDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. l1
    1. process amplitude
    """

    def __init__(self, dx, dt, l2, obs_noise) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.l2 = l2
        self.obs_noise = obs_noise

    def _update_diff_op(self):
        """
        Construct current linearised differential operator.
        """
        l1 = self._params[0]
        partial_t = FinDiff(1, self.dt, 1, acc=2)
        partial_x = FinDiff(0, self.dx, 1, acc=2, periodic=True)
        partial_xxx = FinDiff(0, self.dx, 3, acc=2, periodic=True)
        u0_x = partial_x(self._u0)
        diff_op = partial_t + Coef(l1 * self._u0) * partial_x + Coef(l1 * u0_x) * Identity() + Coef(self.l2) * partial_xxx
        return diff_op

    def _update_prior_precision(self):
        """
        Calculate current prior precision.
        """
        prior_precision = (self.dt * self.dx / self._params[1]**2) * (self._L.T @ self._L)
        return prior_precision

    def _update_prior_mean(self):
        """
        Calculate current prior mean.
        """
        l1 = self._params[0]

        # Construct linearisation remainder term
        partial_x = FinDiff(0, self.dx, 1, acc=2, periodic=True)
        u0_x = partial_x(self._u0)
        remainder = l1 * self._u0 * u0_x

        # Compute prior mean
        prior_mean = spsolve(self._L, remainder.flatten())
        return prior_mean.reshape(self._u0.shape)

    def _update_obs_noise(self):
        """
        Get observation noise (standard deviation).
        """
        return self.obs_noise

dynamics = KdVDynamics(dx, dt, l2, obs_std)

# ######################
# # Fit model with GPR #
# ######################

gp_kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=obs_std**2, noise_level_bounds="fixed")
gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=10)
gp.fit(obs_locs, obs_vals)
test_locs = [[x, t] for x in xx for t in tt]
test_locs_t0 = [[x, 0] for x in xx]
ic_mean, ic_cov = gp.predict(test_locs_t0, return_cov=True)
ic_std = np.sqrt(np.diag(ic_cov))

#################################
# Fit model with iterative INLA #
#################################

def kdv(u, t, L, l1, l2):
    """Differential equations for the KdV equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3)

    # Compute du/dt.    
    dudt = -l1*u*ux - l2*uxxx

    return dudt

# Run model forward with initial condition from GPR and prior mean parameters
# u_guess = odeint(kdv, data_dict['u0_mean'], tt, args=(2, param0[0], l2), mxstep=5000).T
u_guess = odeint(kdv, ic_mean, tt, args=(2, param0[0], l2), mxstep=5000).T

# Fit I-INLA model
model = IterativeINLARegressor(uu, dynamics, param0,
                               u0=u_guess,
                               mixing_coef=0.8,
                               param_bounds=param_bounds,
                               param_priors=param_priors,
                               sampling_evec_scales=[0.1, 0.1],
                               sampling_threshold=3)

model.fit(obs_dict, max_iter=max_iter, parameterisation=parameterisation, animated=True, calc_std=False, calc_mnll=True)

# Save fitted model for further evaluation
results_file = f"results/kdv_{data_id}_{parameterisation}.pkl"
with open(results_file, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

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

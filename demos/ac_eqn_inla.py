import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
from scipy.sparse.linalg import spsolve
from findiff import FinDiff, Coef, Identity

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from spdeinf import util
from spdeinf.nonlinear import SPDEDynamics, IterativeINLARegressor
from spdeinf.distributions import LogNormal

# General configuration
data_id = 4 # 0 - 4
parameterisation = 'moment' # 'moment' or 'natural'
max_iter = 10 # 10 - 20

# Set seed
np.random.seed(0)
# 0:1, 1:3, 2:4, 3:5, 4:6

##################################
# Load data from Allen-Cahn eqn. #
##################################

# Define parameters of the Allen-Cahn eqn. and model
alpha = 0.001
beta = 5
obs_std = 1e-2
params = np.array([alpha, beta, obs_std])

# Define parameters of the model parameter priors
tau_beta = 1
beta_prior_mode = 3.0
beta_0 = np.log(beta_prior_mode) + (tau_beta ** (-2))

# Process noise prior
tau_k = 1
k_prior_mode = 0.01
k_0 = np.log(k_prior_mode) + (tau_k ** (-2))

param0 = np.array([beta_prior_mode, k_prior_mode])
param_priors = [LogNormal(mu=beta_0, sigma=1/tau_beta),
                LogNormal(mu=k_0, sigma=1/tau_k)]
param_bounds = [(1.0, 10.0), (1e-3, 1e-1)]

# fig, ax = plt.subplots(len(param_priors), 1)
# for i, pr in enumerate((param_priors)):
#     domain = np.linspace(*param_bounds[i], 100)
#     ax[i].plot(domain, np.exp(pr.logpdf(domain)))
# plt.show()

# Load Allen-Cahn eqn. data from PINNs examples
data = loadmat("data/PINNs/AC.mat")
uu_full = data['uu']
xx_full = data['x'].squeeze()
tt_full = data['tt'].squeeze()
dx_full = xx_full[1] - xx_full[0]
dt_full = tt_full[1] - tt_full[0]
uu = uu_full[::4,::4]
xx = xx_full[::4]
tt = tt_full[::4]
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]
N_x = xx.shape[0]
N_t = tt.shape[0]
shape = (N_x, N_t)
Xgrid, Tgrid = np.meshgrid(xx, tt, indexing='ij')

# # Sample 256 observations in [0, 0.28]
# obs_count_1 = 256
# obs_loc_1 = np.where(tt == 0.0)[0][0]
# obs_loc_2 = np.where(tt == 0.28)[0][0]
# obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_2+1))
# obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
# obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
# obs_vals = np.array(list(obs_dict.values()), dtype=float)

# Load observations
data_file = f"data/ac_{data_id}.pkl"
with open(data_file, 'rb') as f:
    data_dict = pickle.load(f)
obs_dict = data_dict['obs_dict']
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
obs_vals = np.array(list(obs_dict.values()), dtype=float)

###################################
# Define Allen-Cahn eqn. dynamics #
###################################

class ACDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. beta
    1. process amplitude
    """

    def __init__(self, dx, dt, alpha, obs_noise) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.alpha = alpha
        self.obs_noise = obs_noise

    def _update_diff_op(self):
        """
        Construct current linearised differential operator.
        """
        beta = self._params[0]
        partial_t = FinDiff(1, self.dt, 1, acc=2)
        partial_xx = FinDiff(0, self.dx, 2, acc=2, periodic=True)
        u0_sq = self._u0 ** 2
        diff_op = partial_t - Coef(self.alpha) * partial_xx + Coef(3 * beta * u0_sq) * Identity() - Coef(beta) * Identity()
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
        beta = self._params[0]

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
        return self.obs_noise

dynamics = ACDynamics(dx, dt, alpha, obs_std)

# ######################
# # Fit model with GPR #
# ######################

# gp_kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=obs_std**2, noise_level_bounds="fixed")
# gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=10)
# gp.fit(obs_locs, obs_vals)
# test_locs = [[x, t] for x in xx for t in tt]
# test_locs_t0 = [[x, 0] for x in xx]
# ic_mean, ic_cov = gp.predict(test_locs_t0, return_cov=True)
# ic_std = np.sqrt(np.diag(ic_cov))

# # # Save data for other benchmarks
# obs_table = [[xx[k[0]], tt[k[1]], v] for k, v in obs_dict.items()]
# util.obs_to_csv(obs_table, header="XTU", filename=f"data/ACTrain{data_id}.csv")

# u_table = np.empty((N_x * N_t, 3))
# u_table[:,0] = Xgrid.flatten()
# u_table[:,1] = Tgrid.flatten()
# u_table[:,2] = uu.flatten()
# util.obs_to_csv(u_table, header="XTU", filename=f"data/ACTest.csv")

# data_dict = {'uu': uu, 'uu_full': uu_full, 'xx': xx, 'xx_full': xx_full, 'tt': tt, 'tt_full': tt_full, 'dx': dx, 'dx_full': dx_full, 'dt': dt, 'dt_full': dt_full, 'u0_mean': ic_mean, 'u0_cov': ic_cov, 'u0_std': ic_std, 'obs_dict': obs_dict, 'obs_std': obs_std}
# data_file = f"data/ac_{data_id}.pkl"
# with open(data_file, 'wb') as f:
#     pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

#################################
# Fit model with iterative INLA #
#################################

def ac(u, t, L, alpha, beta):
    """Differential equations for the Allen-Cahn equation, discretized in x."""
    # u_t = alpha*u_xx - beta*(u^3 - u)

    # Compute the x derivatives using the pseudo-spectral method.
    u_xx = psdiff(u, period=L, order=2)

    # Compute du/dt.    
    dudt = alpha*u_xx - beta*(u**3 - u)

    return dudt

# Run model forward with initial condition from GPR and prior mean parameters
# u_guess = odeint(ac, ic_mean, tt, args=(2, alpha, param0[0]), mxstep=5000).T
u_guess = odeint(ac, data_dict['u0_mean'], tt, args=(2, alpha, param0[0]), mxstep=5000).T

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
results_file = f"results/ac_{data_id}_{parameterisation}.pkl"
with open(results_file, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

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
# plt.show()

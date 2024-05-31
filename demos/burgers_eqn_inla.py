import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from findiff import FinDiff, Coef, Identity

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from iinla import util
from iinla.nonlinear import SPDEDynamics, IterativeINLARegressor
from iinla.distributions import LogNormal

# General configuration
data_id = 9 # 0 - 9
parameterisation = 'natural' # 'moment' or 'natural'
max_iter = 10 # 10 - 15

# Set seed
np.random.seed(0)
# 0:1, 1:8, 2:3, 3:9, 4:5, 5:10, 6:11, 7:12, 8:13, 9:15

####################################
# Generate data from Burgers' eqn. #
####################################

# Define parameters of Burgers' equation and model
nu = 0.02 # Kinematic viscosity coefficient
obs_std = 1e-2 # Observation noise
params = np.array([nu, obs_std])

# Define parameters of the model parameter priors
tau_nu = 1
nu_prior_mode = 0.05
nu_0 = np.log(nu_prior_mode) + (tau_nu ** (-2))

# Process noise prior
tau_s = 1
s_prior_mode = 0.01
s_0 = np.log(s_prior_mode) + (tau_s ** (-2))

param0 = np.array([nu_prior_mode, s_prior_mode])
param_priors = [LogNormal(mu=nu_0, sigma=1/tau_nu),
                LogNormal(mu=s_0, sigma=1/tau_s)]
param_bounds = [(0.005, 0.1), (0.00001, 0.1)]

# Create spatial discretisation
L_x = 1                       # Range of spatial domain
dx = 0.04                     # Spatial delta
N_x = int(2 * L_x / dx)       # Number of points in spatial discretisation
xx = np.linspace(-L_x, L_x - dx, N_x)  # Spatial array

# Create temporal discretisation
L_t = 0.5                     # Range of temporal domain
dt_full = 0.001               # Temporal delta
N_t_full = int(L_t / dt_full) # Number of points in temporal discretisation
tt_full = np.linspace(0, L_t - dt_full, N_t_full) # Temporal array

# Define wave number discretisation
k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

# Define the initial condition    
u0 = -np.sin(np.pi * (xx))

def burgers_odes(u, t, k, nu):
    """
    Construct system of ODEs for Burgers' equation using pseudo-spectral method of lines
    """
    # Take x derivatives in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j * k * u_hat
    u_hat_xx = -k**2 * u_hat

    # Transform back to spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)

    # Assemble ODE
    u_t = -u * u_x + nu * u_xx
    return u_t.real

# Solve system of ODEs
uu_full = odeint(burgers_odes, u0, tt_full, args=(k, nu,), mxstep=5000).T

# Downsample domain
dt = 0.02
dt_ratio = round(dt / dt_full)
uu = uu_full[:,::dt_ratio]
tt = tt_full[::dt_ratio]
N_t = len(tt)
shape = (N_x, N_t)
Xgrid, Tgrid = np.meshgrid(xx, tt, indexing='ij')

# # Sample observations
# obs_count_1 = 20
# obs_count_2 = 20
# obs_loc_1 = np.where(tt == 0.0)[0][0]
# obs_loc_2 = np.where(tt == 0.26)[0][0]
# obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_1+1))
# obs_dict.update(util.sample_observations(uu, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
# obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
# obs_vals = np.array(list(obs_dict.values()), dtype=float)
# obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
# print("Number of observations:", obs_idxs.shape[0])

# Load observations
data_file = f"data/burgers_{data_id}.pkl"
with open(data_file, 'rb') as f:
    data_dict = pickle.load(f)
obs_dict = data_dict['obs_dict']
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
obs_vals = np.array(list(obs_dict.values()), dtype=float)

#################################
# Define Burgers' eqn. dynamics #
#################################

class BurgersDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. nu
    1. process amplitude
    """

    def __init__(self, dx, dt, obs_noise) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.obs_noise = obs_noise
        self.partial_t = FinDiff(1, dt, 1, acc=2)
        self.partial_x = FinDiff(0, dx, 1, acc=2, periodic=True)
        self.partial_xx = FinDiff(0, dx, 2, acc=2, periodic=True)

    def _update_diff_op(self):
        """
        Construct current linearised differential operator.
        """
        nu = self._params[0]
        u0_x = self.partial_x(self._u0)
        diff_op = self.partial_t + Coef(self._u0) * self.partial_x - Coef(nu) * self.partial_xx + Coef(u0_x) * Identity()
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
        # Construct linearisation remainder term
        u0_x = self.partial_x(self._u0)
        remainder = self._u0 * u0_x

        # Compute prior mean
        prior_mean = spsolve(self._L.T @ self._L, self._L.T @ remainder.flatten())
        return prior_mean.reshape(self._u0.shape)

    def _update_obs_noise(self):
        """
        Get observation noise (standard deviation).
        """
        return self.obs_noise

dynamics = BurgersDynamics(dx, dt, obs_std)

######################
# Fit model with GPR #
######################

# gp_kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=obs_std**2, noise_level_bounds="fixed")
# gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=10)
# gp.fit(obs_locs, obs_vals)
# test_locs = [[x, t] for x in xx for t in tt]
# test_locs_t0 = [[x, 0] for x in xx]
# ic_mean, ic_cov = gp.predict(test_locs_t0, return_cov=True)
# ic_std = np.sqrt(np.diag(ic_cov))

# # Save data for other benchmarks
# obs_table = [[xx[k[0]], tt[k[1]], v] for k, v in obs_dict.items()]
# util.obs_to_csv(obs_table, header="XTU", filename=f"data/BurgersTrain{data_id}.csv")

# u_table = np.empty((N_x * N_t, 3))
# u_table[:,0] = Xgrid.flatten()
# u_table[:,1] = Tgrid.flatten()
# u_table[:,2] = uu.flatten()
# util.obs_to_csv(u_table, header="XTU", filename=f"data/BurgersTest.csv")

# data_dict = {'uu': uu, 'uu_full': uu_full, 'xx': xx, 'tt': tt, 'tt_full': tt_full, 'dx': dx, 'dt': dt, 'dt_full': dt_full, 'u0_mean': ic_mean, 'u0_cov': ic_cov, 'u0_std': ic_std, 'obs_dict': obs_dict, 'obs_std': obs_std}
# data_file = f"data/burgers_{data_id}.pkl"
# with open(data_file, 'wb') as f:
#     pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

##########################################
# Fit model with iterative linearisation #
##########################################

# Run model forward with initial condition from GPR and prior mean parameters
# u_guess = odeint(burgers_odes, ic_mean, tt, args=(k, param0[0],), mxstep=5000).T
u_guess = odeint(burgers_odes, data_dict['u0_mean'], tt, args=(k, param0[0],), mxstep=5000).T

# Fit I-INLA model
model = IterativeINLARegressor(uu, dynamics, param0,
                               u0=u_guess,
                               mixing_coef=0.8,
                               param_bounds=param_bounds,
                               param_priors=param_priors,
                               sampling_evec_scales=[0.1, 0.1],
                               sampling_threshold=5)
model.fit(obs_dict, max_iter=max_iter, parameterisation=parameterisation, animated=True, calc_std=True, calc_mnll=True)

# Save fitted model for further evaluation
results_file = f"results/burgers_{data_id}_{parameterisation}.pkl"
with open(results_file, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

############
# Plot fit #
############

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_mean, extent=(0, .5, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="RdBu_r")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/burgers_eqn/burgers_spde_inla.pdf", transparent=True)

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_std, extent=(0, .5, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/burgers_eqn/burgers_spde_inla_std.pdf", transparent=True)
plt.show()

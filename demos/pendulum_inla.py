import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from findiff import FinDiff, Coef, Identity

from spdeinf import util
from spdeinf.nonlinear import SPDEDynamics, IterativeINLARegressor
from spdeinf.distributions import LogNormal

# Set seed
np.random.seed(0)

#####################################################
# Generate data from nonlinear damped pendulum eqn. #
#####################################################

# Define parameters of damped pendulum
b = 0.3
c = 1.
params_true = np.array([b, c])
print("True parameters:", params_true)

# Define parameters of the model parameter priors
tau_b = 2
b_prior_mode = 0.2
b_0 = np.log(b_prior_mode) + (tau_b ** (-2))

tau_c = 1
c_prior_mode = 2.
c_0 = np.log(c_prior_mode) + (tau_c ** (-2))

# Process noise prior
tau_k = 2
k_prior_mode = 0.05
k_0 = np.log(k_prior_mode) + (tau_k ** (-2))

# Observation noise prior
tau_s = 2
s_prior_mode = 0.2
s_0 = np.log(s_prior_mode) + (tau_s ** (-2))

param0 = np.array([b_prior_mode, c_prior_mode, k_prior_mode, s_prior_mode])
param_priors = [LogNormal(mu=b_0, sigma=1/tau_b), LogNormal(mu=c_0, sigma=1/tau_c),
                LogNormal(mu=k_0, sigma=1/tau_k), LogNormal(mu=s_0, sigma=1/tau_s)]
param_bounds = [(0.1, 1), (0.1, 5), (0, 1), (0, 1)]

# Create temporal discretisation
L_t = 25                      # Duration of simulation [s]
dt = 0.05                     # Infinitesimal time
N_t = int(L_t / dt) + 1       # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array
# T = np.around(T, decimals=1)

# Define the initial condition
u0 = [0.75 * np.pi, 0.]

# Define corresponding system of ODEs
def pend(u, t, b, c):
    theta, omega = u
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

# Solve system of ODEs
u = odeint(pend, u0, T, args=(b, c,))

# For our purposes we only need the solution for the pendulum angle, theta
u = u[:, 0].reshape(1, -1)

# Sample observations
obs_std = 1e-1
obs_count = 20
obs_loc_1 = np.where(T == 5.)[0][0]
obs_dict = util.sample_observations(u, obs_count, obs_std, extent=(None, None, 0, obs_loc_1))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()))
print("Number of observations:", obs_idxs.shape[0])

######################################
# Define nonlinear pendulum dynamics #
######################################

class PendulumDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. b
    1. c
    2. process amplitude
    3. observation noise
    """

    def __init__(self, dt) -> None:
        super().__init__()
        self.dt = dt

    def _update_diff_op(self):
        """
        Constructs linearised differential operator based on current state.
        """
        b = self._params[0]
        c = self._params[1]
        partial_t = FinDiff(1, self.dt, 1)
        partial_tt = FinDiff(1, self.dt, 2)
        u0_cos = np.cos(self._u0)
        diff_op = partial_tt + Coef(b) * partial_t + Coef(c * u0_cos) * Identity()
        return diff_op

    def _update_prior_precision(self):
        """
        Calculate current prior precision.
        """
        prior_precision = (self.dt / self._params[2]**2) * (self._L.T @ self._L)
        return prior_precision

    def _update_prior_mean(self):
        """
        Calculate current prior mean.
        """
        c = self._params[1]
        u0_cos = np.cos(self._u0)
        u0_sin = np.sin(self._u0)
        prior_mean = spsolve(self._L, (c * (self._u0 * u0_cos - u0_sin)).flatten())
        return prior_mean.reshape(self._u0.shape)

    def _update_obs_noise(self):
        """
        Get observation noise (standard deviation).
        """
        return self._params[3]

dynamics = PendulumDynamics(dt)

#################################
# Fit model with iterative INLA #
#################################

max_iter = 5
parameterisation = 'natural' # 'moment' or 'natural'
model = IterativeINLARegressor(u, dynamics, param0,
                               mixing_coef=0.3,
                               param_bounds=param_bounds,
                               param_priors=param_priors,
                               sampling_evec_scales=[0.05, 0.05, 0.05, 0.05],
                               sampling_threshold=2.5)

model.fit(obs_dict, max_iter=max_iter, parameterisation=parameterisation, animated=False, calc_std=False, calc_mnll=True)

############
# Plot fit #
############

num_samples = 5000
final_dist = model.marginal_dist_u_y
samples = final_dist.sample(num_samples)
samples = np.reshape(samples, (num_samples, -1))

weights = np.ones(num_samples) / num_samples
creds = [50, 60, 70, 80, 90, 95]

list_of_intervals = []
for i in range(samples.shape[1]):
    intervals = util.cred_wt(samples[:,i], weights, creds)
    list_of_intervals.append(intervals)

plt.figure(figsize=(4,3))
for cred in creds:
    lower_lims = [interval[cred][0] for interval in list_of_intervals]
    upper_lims = [interval[cred][1] for interval in list_of_intervals]
    plt.fill_between(T, lower_lims, upper_lims, color='steelblue', alpha=1-cred/120, edgecolor='None')

plt.plot(T, u[0], c='C1', linewidth=4)
plt.plot(T, model.u0[0], c='k', linestyle='--', linewidth=2)
plt.axvline(obs_loc_1 * dt, color='grey', ls=':', linewidth=2) 
plt.scatter(obs_idxs[:,1] * dt, obs_vals, c='k', zorder=10)
plt.xlabel("$t$", fontsize=16)
plt.ylabel("$u$", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("figures/pendulum/pendulum_sode_inla.png", dpi=200)
plt.tight_layout()
# plt.show()

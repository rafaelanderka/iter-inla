#%%
import os
import pickle
import sdeint
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint 
from sksparse.cholmod import cholesky
from findiff import FinDiff, Coef, Identity
from spdeinf import util
from spdeinf.nonlinear import SPDEDynamics, IterativeINLARegressor
from spdeinf.distributions import LogNormal
from functools import partial
import matplotlib.font_manager as font_manager

package_dir = '/Users/sotakao/Dropbox/Mac/Documents/spde-inference'
os.chdir(package_dir)

#####################################################
# Generate data from nonlinear damped pendulum eqn. #
#####################################################

generate_data = True # Flag to generate observations or load externally
just_plot = False # Flag to train from scratch or to load saved result for plotting

## Generate data from stochastic nonlinear damped pendulum eqn.
# Define parameters of damped pendulum
b = 0.3
c = 1.
s = 0.3

# Define parameters of the model parameter priors
tau_b = 2
b_prior_mode = 0.2
b_0 = np.log(b_prior_mode) + (tau_b ** (-2))

tau_c = 1
c_prior_mode = 2.
c_0 = np.log(c_prior_mode) + (tau_c ** (-2))

# Process noise prior
tau_s = 2
s_prior_mode = 0.1
s_0 = np.log(s_prior_mode) + (tau_s ** (-2))

# Observation noise prior
tau_t = 2
t_prior_mode = 0.1
t_0 = np.log(t_prior_mode) + (tau_t ** (-2))

param0 = np.array([b_prior_mode, c_prior_mode, s_prior_mode, t_prior_mode])
param_priors = [LogNormal(mu=b_0, sigma=1/tau_b), LogNormal(mu=c_0, sigma=1/tau_c),
                LogNormal(mu=s_0, sigma=1/tau_s), LogNormal(mu=t_0, sigma=1/tau_t)]
param_bounds = [(0.1, 1), (0.1, 5), (0, 2), (0, 1)]

# Create temporal discretisation
L_t = 25                     # Duration of simulation [s]
dt = 0.01                      # Infinitesimal time
N_t = int(L_t / dt) + 1       # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array
# T = np.around(T, decimals=1)

if generate_data:
    # Define the initial condition    
    u0 = [0.75 * np.pi, 0.]

    # Define drift and diffusion of SDE
    SEED = 2374

    def pend(u, t, b, c):
        theta, omega = u
        dydt = np.array([omega, -b*omega - c*np.sin(theta)])
        return dydt

    def diffusion(x, t):
        B = np.diag([0., s**2])
        return B

    drift = partial(pend, b=b, c=c)

    # Solve system of Ito SDE
    u = sdeint.itoSRI2(drift, diffusion, u0, T, generator=np.random.default_rng(SEED))

    # For our purposes we only need the solution for the pendulum angle, theta
    u = u[:, 0].reshape(1, -1)

    # Sample observations
    obs_std = 1e-1
    obs_count = 60
    obs_loc_1 = np.where(T == 10.)[0][0]
    obs_dict = util.sample_observations(u, obs_count, obs_std, extent=(None, None, 0, obs_loc_1))
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    idxs = obs_idxs[:,1]
    obs_vals = np.array(list(obs_dict.values()))
    print("Number of observations:", obs_idxs.shape[0])
else:
    # Load pre-generated data
    cwd = os.getcwd()
    fname = 'demos/pf_baselines/stoch_pend_b_0.3_c_1_sigmaX_0.2_sigmaY_0.1.npz'

    data = np.load(os.path.join(cwd, fname))
    u = data['x'][:,0].reshape(1, -1)
    idxs = np.where(~np.isnan(data['y']))[0]
    obs_vals = data['y'][idxs]
    obs_idxs = [(0, idx) for idx in idxs]
    obs_dict = dict(zip(obs_idxs, obs_vals))

    # Variables for plotting
    T = np.linspace(0, L_t, u.shape[1]) 
    T = np.around(T, decimals=1)
    obs_loc_1 = np.where(T == 10.)[0][0]


#%%
#############################################
# Set up nonlinear pendulum regressor class #
#############################################

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

    def get_diff_op(self, u0, params, **kwargs):
        """
        Construct current linearised differential operator.
        """
        b, c, _, _ = params
        partial_t = FinDiff(1, self.dt, 1)
        partial_tt = FinDiff(1, self.dt, 2)
        u0_cos = np.cos(u0)
        diff_op = partial_tt + Coef(b) * partial_t + Coef(c * u0_cos) * Identity()
        return diff_op

    def get_prior_precision(self, u0, params, **kwargs):
        """
        Calculate current prior precision.
        """
        diff_op_guess = self.get_diff_op(u0, params)
        L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
        prior_precision = (self.dt / params[2]**2) * (L.T @ L)
        return prior_precision

    def get_prior_mean(self, u0, params, **kwargs):
        """
        Calculate current prior mean.
        """
        b, c, _, _ = params
        u0_cos = np.cos(u0)
        u0_sin = np.sin(u0)
        diff_op = self.get_diff_op(u0, params)
        diff_op_mat = diff_op.matrix(u0.shape)
        prior_mean = spsolve(diff_op_mat, (c * (u0 * u0_cos - u0_sin)).flatten())
        return prior_mean.reshape(u0.shape)

    def get_obs_noise(self, params, **kwargs):
        """
        Get observation noise (standard deviation).
        """
        # return self.obs_std
        return params[3]

    
dynamics = PendulumDynamics(dt)

############################################
# Fit Nonlinear Pendulum Regressor on data #
############################################
max_iter = 20
parameterisation = 'natural' # 'moment' or 'natural'
model = IterativeINLARegressor(u, dynamics, param0,
                               mixing_coef=0.2,
                               param_bounds=param_bounds,
                               param_priors=param_priors,
                               sampling_evec_scales=[0.03, 0.03, 0.03, 0.03],
                               sampling_threshold=5)

if just_plot is False:
    model.fit(obs_dict, obs_std=1e-1, max_iter=max_iter, parameterisation=parameterisation, animated=False, calc_std=False, calc_mnll=True)
    iter_count = len(model.mse_hist)

    # Save result
    with open('results/stoch_pendulum_iterated_inla.pickle', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('results/stoch_pendulum_iterated_inla.pickle', "rb") as f:
        model = pickle.load(f)

#%%
############
# Plot fit #
############
plt.figure(figsize=(3,3))
plt.plot(T, u.squeeze(), "b", label="Ground truth")
plt.plot(T, model.posterior_mean.squeeze(), "k", label="Posterior mean")
plt.plot(T, model.posterior_mean.squeeze() + model.posterior_std.squeeze(), "--", color="grey", label="Posterior std. dev.")
plt.plot(T, model.posterior_mean.squeeze() - model.posterior_std.squeeze(), "--", color="grey")
plt.axvline(dt * obs_loc_1, color='grey', ls=':')
plt.scatter(dt * idxs, obs_vals, c="r", marker="x", label="Observations")
plt.xlabel("$t$")
plt.ylabel("$u$")
plt.tight_layout()
# plt.savefig("figures/pendulum/pendulum_spde_inla_new_fit.pdf")
plt.show()

# # Save animation
# print("Saving animation...")
# model.save_animation("figures/pendulum/pendulum_inla_new_iter_animation.gif", fps=3)


# %%
from spdeinf.util import cred_wt

# # Compute MMD score
# fname = 'demos/pf_baselines/state_posterior_samples_3.npy'
# posterior_samples = np.load(fname) # Get state samples from the particle smoother
# posterior_samples = posterior_samples[:,:,0]
# N = posterior_samples.shape[1] # Number of samples

# # shuffle sample indices for each time step for consistent MMD evaluation
# for t, samples_t in enumerate(posterior_samples):
#     shuffled_idxs = np.random.choice(N, size=(N,), replace=False)
#     posterior_samples[t] = posterior_samples[t][shuffled_idxs]

num_samples = 5000
final_dist = model.marginal_dist_u_y
samples = final_dist.sample(num_samples)
samples = np.reshape(samples, (num_samples, -1))

# print("Computing MMD... (this takes a minute or two to compute)")
# x = posterior_samples.T
# y = samples
# mmd = util.MMD(x, y)

# print(f"MMD = {mmd}")

weights = np.ones(num_samples) / num_samples
creds = [50, 60, 70, 80, 90, 95]

list_of_intervals = []
for i in range(samples.shape[1]):
    intervals = cred_wt(samples[:,i], weights, creds)
    list_of_intervals.append(intervals)

#%%
plt.figure(figsize=(4,3))
for cred in creds:
    lower_lims = [interval[cred][0] for interval in list_of_intervals]
    upper_lims = [interval[cred][1] for interval in list_of_intervals]
    plt.fill_between(T, lower_lims, upper_lims, color='steelblue', alpha=1-cred/120, edgecolor='None')

plt.plot(T, u[0], c='C1', linewidth=4)
# plt.plot(T, model.u0[0], c='k', linestyle='--', linewidth=2)
plt.axvline(obs_loc_1 * dt, color='grey', ls=':', linewidth=2) 
plt.scatter(idxs * dt, obs_vals, c='k', zorder=10)
# font = font_manager.FontProperties(family='Times New Roman',
#                                    style='normal', size=16,
#                                    weight='bold')
# plt.text(13, 2, f"MMD = {mmd:.3f}", fontproperties=font)
plt.xlabel("$t$", fontsize=12)
plt.ylabel("$u$", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
package_dir = '/Users/sotakao/Dropbox/Mac/Documents/spde-inference'
plt.savefig(os.path.join(package_dir, "figures/pendulum/stoch_pendulum_spde_inla_fit.pdf"))
plt.show()


# %%

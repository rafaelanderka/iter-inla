#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from sksparse.cholmod import cholesky
from findiff import FinDiff, Coef, Identity
from spdeinf import util
from spdeinf.nonlinear import AbstractNonlinearINLASPDERegressor
from spdeinf.distributions import LogNormal

#####################################################
# Generate data from nonlinear damped pendulum eqn. #
#####################################################

# Define parameters of damped pendulum
b = 0.3
c = 1.
params_true = np.array([b, c])

# Define parameters of the model parameter priors
tau_b = 1
b_prior_mode = 0.2
b_0 = np.log(b_prior_mode) + (tau_b ** (-2))

tau_c = 1
c_prior_mode = 2.
c_0 = np.log(c_prior_mode) + (tau_c ** (-2))

# Process noise prior
tau_k = 1
k_prior_mode = 0.05
k_0 = np.log(k_prior_mode) + (tau_k ** (-2))

# Observation noise prior
tau_s = 1
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
T = np.around(T, decimals=1)

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


#############################################
# Set up nonlinear pendulum regressor class #
#############################################

class NonlinearPendulumINLARegressor(AbstractNonlinearINLASPDERegressor):
    """
    The parameters of the model are, in order:
    0. b
    1. c
    2. process amplitude
    3. observation noise
    """

    def _get_diff_op(self, u0, params, **kwargs):
        """
        Construct current linearised differential operator.
        """
        b, c, _, _ = params
        partial_t = FinDiff(1, self.dt, 1)
        partial_tt = FinDiff(1, self.dt, 2)
        u0_cos = np.cos(u0)
        diff_op = partial_tt + Coef(b) * partial_t + Coef(c * u0_cos) * Identity()
        return diff_op

    def _get_prior_precision(self, u0, params, **kwargs):
        """
        Calculate current prior precision.
        """
        diff_op_guess = self._get_diff_op(u0, params)
        L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
        prior_precision = (self.dt / params[2]**2) * (L.T @ L)
        return prior_precision

    def _get_prior_mean(self, u0, params, **kwargs):
        """
        Calculate current prior mean.
        """
        b, c, _, _ = params
        u0_cos = np.cos(u0)
        u0_sin = np.sin(u0)
        diff_op = self._get_diff_op(u0, params)
        diff_op_mat = diff_op.matrix(u0.shape)
        prior_mean = spsolve(diff_op_mat, (c * (u0 * u0_cos - u0_sin)).flatten())
        return prior_mean.reshape(u0.shape)

    def _get_obs_noise(self, params, **kwargs):
        """
        Get observation noise (standard deviation).
        """
        # return self.obs_std
        return params[3]

    
############################################
# Fit Nonlinear Pendulum Regressor on data #
############################################
max_iter = 10
parameterisation = 'natural' # 'moment' or 'natural'
model = NonlinearPendulumINLARegressor(u, 1, dt, param0,
                                       mixing_coef=0.5,
                                       param_bounds=param_bounds,
                                       param_priors=param_priors,
                                       sampling_evec_scales=[0.1, 0.1, 0.1, 0.02],
                                       sampling_threshold=1)

model.fit(obs_dict, obs_std, max_iter=max_iter, parameterisation=parameterisation, animated=True, calc_std=True, calc_mnll=True)
iter_count = len(model.mse_hist)


############
# Plot fit #
############
plt.figure(figsize=(3,3))
plt.plot(T, u.squeeze(), "b", label="Ground truth")
plt.plot(T, model.posterior_mean.squeeze(), "k", label="Posterior mean")
plt.plot(T, model.posterior_mean.squeeze() + model.posterior_std.squeeze(), "--", color="grey", label="Posterior std. dev.")
plt.plot(T, model.posterior_mean.squeeze() - model.posterior_std.squeeze(), "--", color="grey")
plt.axvline(dt * obs_loc_1, color='grey', ls=':')
plt.scatter(dt * obs_idxs[:,1], obs_vals, c="r", marker="x", label="Observations")
plt.xlabel("$t$")
plt.ylabel("$u$")
plt.tight_layout()
plt.savefig("figures/pendulum/pendulum_spde_inla_new_fit.pdf")
plt.show()

# Save animation
print("Saving animation...")
model.save_animation("figures/pendulum/pendulum_inla_new_iter_animation.gif", fps=3)



# %%

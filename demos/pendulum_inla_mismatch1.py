import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sksparse.cholmod import cholesky
from findiff import FinDiff, Coef, Identity

from spdeinf import linear, nonlinear, util

## Generate data from nonlinear damped pendulum eqn.

# Define parameters of damped pendulum
b = 0.3
c = 1.
params_true = np.array([b, c])

# Define parameters of the parameter priors
tau_b = 1
b_prior_mode = 0.2
b_0 = np.log(b_prior_mode) + (tau_b ** (-2))
tau_c = 1
c_prior_mode = 2.
c_0 = np.log(c_prior_mode) + (tau_c ** (-2))
print(b_prior_mode, c_prior_mode)
# param0 = np.array([b_prior_mode, c_prior_mode, 1, 1e-1])
# param0 = np.array([b_prior_mode, c_prior_mode])
param0 = np.array([b_prior_mode, 1])
# param_bounds = [(0.1, 1), (0.1, 15), (1, 50), (1e-4, 1)]
# param_bounds = [(0.1, 1), (0.1, 15)]
param_bounds = [(0.1, 1), (1, 10)]

# Create temporal discretisation
L_t = 25                      # Duration of simulation [s]
dt = 0.05                     # Infinitesimal time
N_t = int(L_t / dt) + 1       # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array
T = np.around(T, decimals=2)

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

## Generate GP noise
# Define the Gaussian Process with RBF kernel
gp_noise_level = 0.5
kernel = 1.0 * RBF(length_scale=1)
gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
y_gp = gp.sample_y(T.reshape(-1,1), 1).squeeze()

# Add noise to simulation
u = u + gp_noise_level * y_gp

# Plot the function, the prediction and the 95% confidence interval based on the MSE
fig, ax = plt.subplots(1, 1, figsize=(5,3))
ax.plot(T, y_gp, 'k', lw=1)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
plt.show()

# Sample observations
obs_std = 1e-1
obs_count = 20
obs_loc_1 = np.where(T == 5.)[0][0]
obs_dict = util.sample_observations(u, obs_count, obs_std, extent=(None, None, 0, obs_loc_1))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()))
print("Number of observations:", obs_idxs.shape[0])

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dt, params):
    """
    Constructs current linearised differential operator.
    """
    # b, c, _, _ = params
    # b, c = params
    b, _ = params
    partial_t = FinDiff(1, dt, 1)
    partial_tt = FinDiff(1, dt, 2)
    u0_cos = np.cos(u0)
    diff_op = partial_tt + Coef(b) * partial_t + Coef(c * u0_cos) * Identity()
    return diff_op

def get_prior_mean(u0, diff_op_gen, params):
    """
    Calculates current prior mean.
    """
    # b, c, _, _ = params
    # b, c = params
    b, _ = params
    u0_cos = np.cos(u0)
    u0_sin = np.sin(u0)
    diff_op = diff_op_gen(u0, params)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (c * (u0 * u0_cos - u0_sin)).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u, params: get_diff_op(u, dt, params)
prior_mean_gen = lambda u, params: get_prior_mean(u, diff_op_gen, params)

##################################
# log-pdf of parameter posterior #
##################################

def _logpdf_marginal_posterior(b, c, Q_u, Q_uy, Q_obs, mu_u, mu_uy, obs_dict, shape, regularisation=1e-3):
    # Unpack parameters
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    obs_idxs_flat = shape[1] * obs_idxs[:,0] + obs_idxs[:,1]
    obs_vals = np.array(list(obs_dict.values()), dtype=float)

    # Define reused consts.
    log_2pi = np.log(2 * np.pi)
    N = Q_obs.shape[0]
    M = Q_u.shape[0]

    # Perform matrix factorisation to compute log determinants
    Q_u_chol = cholesky(Q_u + regularisation * sparse.identity(Q_u.shape[0]))
    Q_u_logdet = Q_u_chol.logdet()
    Q_uy_chol = cholesky(Q_uy)
    Q_uy_logdet = Q_uy_chol.logdet()
    Q_obs_chol = cholesky(Q_obs)
    Q_obs_logdet = Q_obs_chol.logdet()

    # Compute b prior terms
    if b > 0:
        log_b = np.log(b)
        log_p_b = np.log(tau_b) - log_b - 0.5 * ((tau_b * (log_b - b_0)) ** 2 + log_2pi) # log-normal prior
    else:
        log_p_b = float("-inf")

    # Compute c prior terms
    if c > 0:
        log_c = np.log(c)
        log_p_c = np.log(tau_c) - log_c - 0.5 * ((tau_c * (log_c - c_0)) ** 2 + log_2pi) # log-normal prior
    else:
        log_p_c = float("-inf")

    # Compute GMRF prior terms
    diff_mu_uy_mu_u = mu_uy - mu_u
    log_p_ut = 0.5 * (Q_u_logdet - diff_mu_uy_mu_u.T @ Q_u @ diff_mu_uy_mu_u - M * log_2pi)

    # Compute obs model terms
    diff_obs_mu_uy = obs_vals - mu_uy[obs_idxs_flat]
    log_p_yut = 0.5 * (Q_obs_logdet - diff_obs_mu_uy.T @ Q_obs @ diff_obs_mu_uy - N * log_2pi)

    # Compute full conditional terms
    log_p_uyt = 0.5 * (Q_uy_logdet - M * log_2pi)

    logpdf = log_p_b + log_p_c + log_p_ut + log_p_yut - log_p_uyt
    return logpdf

def logpdf_marginal_posterior(x, u0, obs_dict, diff_op_gen, prior_mean_gen, return_conditional_params=False, debug=False):
    # Process args
    obs_count = len(obs_dict.keys())

    # Compute prior mean
    prior_mean = prior_mean_gen(u0, x)

    # Construct precision matrix corresponding to the linear differential operator
    diff_op_guess = diff_op_gen(u0, x)
    L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
    # prior_precision = x[2] * (L.T @ L)
    prior_precision = x[1] * (L.T @ L)
    # prior_precision = L.T @ L

    # Get "data term" of full conditional
    res = linear._fit_gmrf(u, obs_dict, obs_std, prior_mean, prior_precision, calc_std=return_conditional_params,
                         include_initial_cond=False, return_posterior_precision=True, regularisation=1e-5)

    # Define prior and full conditional params
    mu_u = prior_mean
    Q_u = prior_precision
    mu_uy = res['posterior_mean']
    Q_uy = res['posterior_precision']
    Q_obs = sparse.diags([obs_std ** (-2)], 0, shape=(obs_count, obs_count), format='csc')

    if debug:
        plt.figure()
        plt.plot(prior_mean[0])
        plt.show()

    # Compute marginal posterior
    logpdf = _logpdf_marginal_posterior(x[0], x[1], Q_u, Q_uy, Q_obs, mu_u.flatten(), mu_uy.flatten(), obs_dict, u0.shape)
    # logpdf = _logpdf_marginal_posterior(x[0], c, Q_u, Q_uy, Q_obs, mu_u.flatten(), mu_uy.flatten(), obs_dict, u0.shape)
    if return_conditional_params:
        return logpdf, mu_uy, res['posterior_var']
    return logpdf


## Fit GP with linearised SODE prior from nonlinear pendulum eqn.

# Perform iterative optimisation
max_iter = 10
model = nonlinear.NonlinearINLASPDERegressor(u, 1, dt, param0, diff_op_gen, prior_mean_gen, logpdf_marginal_posterior,
                                             mixing_coef=0.5, param_bounds=param_bounds, sampling_evec_scales=[0.1, 0.05],
                                             sampling_threshold=1)
# model = nonlinear.NonlinearINLASPDERegressor(u, 1, dt, param0, diff_op_gen, prior_mean_gen, logpdf_marginal_posterior,
#                                              mixing_coef=0.5, param_bounds=param_bounds, sampling_evec_scales=[0.1, 0.05, 0.1, 0.1],
#                                              sampling_threshold=1)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)
iter_count = len(model.mse_hist)

# Plot fit
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
plt.savefig("figures/pendulum/pendulum_spde_inla_fit2.pdf")
plt.show()

# Save animation
print("Saving animation...")
model.save_animation("figures/pendulum/pendulum_inla_iter_animation4.gif", fps=3)

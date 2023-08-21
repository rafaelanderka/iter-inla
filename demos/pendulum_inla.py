import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from sksparse.cholmod import cholesky
from findiff import FinDiff, Coef, Identity

from spdeinf import linear, nonlinear, util

## Generate data from nonlinear damped pendulum eqn.

# Define parameters of damped pendulum
b = 0.3
c = 10.

# Define parameters of the parameter priors
tau_b = 10
b_0 = -1.1
b_prior_mode = np.exp(b_0 - tau_b ** (-2))
tau_c = 10
c_0 = 1.5
c_prior_mode = np.exp(c_0 - tau_c ** (-2))
print(b_prior_mode, c_prior_mode)

# Create temporal discretisation
L_t = 25                      # Duration of simulation [s]
dt = 0.1                      # Infinitesimal time
N_t = int(L_t / dt) + 1       # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array

# Define the initial condition    
u0 = [np.pi - 0.1, 0.]

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
obs_std = 1e-4
obs_count = 30
obs_lim = 50
obs_dict = util.sample_observations(u, obs_count, obs_std, extent=(None, None, 0, obs_lim))
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
    b, c = params
    partial_t = FinDiff(1, dt, 1)
    partial_tt = FinDiff(1, dt, 2)
    u0_cos = np.cos(u0)
    diff_op = partial_tt + Coef(b) * partial_t + Coef(c * u0_cos) * Identity()
    return diff_op

def get_prior_mean(u0, diff_op_gen, params):
    """
    Calculates current prior mean.
    """
    b, c = params
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
    obs_vals = np.array(list(obs_dict.values()))

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

    # # Compute obs noise prior terms
    # if t_obs >= 0:
    #     log_t_obs = np.log(t_obs)
    #     log_p_t_obs = (t_obs_a - 1) * log_t_obs - t_obs_b * t_obs + t_obs_a * np.log(t_obs_b) - loggamma(t_obs_a)
    # else:
    #     log_p_t_obs = float("-inf")

    # Compute GMRF prior terms
    diff_mu_uy_mu_u = mu_uy - mu_u
    log_p_ut = 0.5 * (Q_u_logdet - diff_mu_uy_mu_u.T @ Q_u @ diff_mu_uy_mu_u - M * log_2pi)

    # Compute obs model terms
    diff_obs_mu_uy = obs_vals - mu_uy[obs_idxs_flat]
    log_p_yut = 0.5 * (Q_obs_logdet - diff_obs_mu_uy.T @ Q_obs @ diff_obs_mu_uy - N * log_2pi)

    # Compute full conditional terms
    log_p_uyt = 0.5 * (Q_uy_logdet - M * log_2pi)

    # print(log_p_b, log_p_c, log_p_ut, log_p_yut, log_p_uyt)
    logpdf = log_p_b + log_p_c + log_p_ut + log_p_yut - log_p_uyt
    # logpdf = log_p_b + log_p_c
    # logpdf = log_p_ut
    # logpdf = log_p_b + log_p_t_obs
    return logpdf

def logpdf_marginal_posterior(x, u0, obs_dict, diff_op_gen, prior_mean_gen, return_conditional_params=False, debug=False):
    # Process args
    obs_count = len(obs_dict.keys())

    # Compute prior mean
    prior_mean = prior_mean_gen(u0, x)

    # Construct precision matrix corresponding to the linear differential operator
    diff_op_guess = diff_op_gen(u0, x)
    L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
    LL = L.T @ L
    # LL_chol = cholesky(LL + tol * identity(LL.shape[0]))
    # kappa = LL_chol.spinv().diagonal().mean()
    # prior_precision = (self.sigma ** 2) / (self.dV * kappa) * LL
    # prior_precision = (self.sigma ** 2) / (self.dV) * LL
    prior_precision = LL

    # Get "data term" of full conditional
    res = linear._fit_gp(u, obs_dict, obs_std, prior_mean, prior_precision, calc_std=return_conditional_params, calc_lml=False,
                         include_initial_cond=False, return_posterior_precision=True, regularisation=1e-5)

    # Define prior and full condtional params
    mu_u = prior_mean
    Q_u = prior_precision
    mu_uy = res['posterior_mean']
    # mu_uy = u # For testing
    Q_uy = res['posterior_precision']
    Q_obs = sparse.diags([obs_std ** (-2)], 0, shape=(obs_count, obs_count), format='csc')

    if debug:
        plt.figure()
        plt.plot(prior_mean[0])
        plt.show()

    # Compute marginal posterior
    logpdf = _logpdf_marginal_posterior(x[0], x[1], Q_u, Q_uy, Q_obs, mu_u.flatten(), mu_uy.flatten(), obs_dict, u0.shape)
    if return_conditional_params:
        return logpdf, mu_uy, res['posterior_var']
    return logpdf


## Fit GP with non-linear SPDE prior from Burgers' equation

# Perform iterative optimisation
max_iter = 15
model = nonlinear.NonlinearINLASPDERegressor(u, 1, dt, diff_op_gen, prior_mean_gen, logpdf_marginal_posterior, mixing_coef=0.5)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True)
iter_count = len(model.mse_hist)

# Save animation
print("Saving animation...")
model.save_animation("figures/pendulum/pendulum_inla_iter_animation.gif", fps=3)
plt.show()

# Plot fit
plt.plot(T, u.squeeze(), "b", label="Ground truth")
plt.plot(T, model.posterior_mean.squeeze(), "k", label="Posterior mean")
plt.plot(T, model.posterior_mean.squeeze() + model.posterior_std.squeeze(), "--", color="grey", label="Posterior std.")
plt.plot(T, model.posterior_mean.squeeze() - model.posterior_std.squeeze(), "--", color="grey")
plt.axvline(dt * obs_lim, color='grey', ls=':')
plt.scatter(dt * obs_idxs[:,1], obs_vals, c="r", marker="x", label="Observations")
plt.legend(loc="best")
plt.xlabel("t")
plt.ylabel("$\\theta$")
plt.savefig("figures/pendulum/pendulum_fit.png", dpi=200)
plt.show()

# Plot convergence history
plt.figure()
plt.plot(np.arange(1, iter_count + 1), model.mse_hist, label="Linearisation via expansion")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.xticks(np.arange(2, iter_count + 1, 2))
plt.legend()
plt.savefig("figures/pendulum/mse_conv.png", dpi=200)
plt.show()

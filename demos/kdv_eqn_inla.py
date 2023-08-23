import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky
from findiff import FinDiff, Coef, Identity

from spdeinf import linear, nonlinear, util

# Define true parameters of the Korteweg-de Vries eqn.
l1 = 1
l2 = 0.0025
params_true = np.array([l1, l2])

# Define parameters of the parameter priors
tau_l1 = 10
l1_0 = 0
l1_prior_mode = np.exp(l1_0 - tau_l1 ** (-2))

tau_l2 = 10
l2_0 = -6
l2_prior_mode = np.exp(l2_0 - tau_l2 ** (-2))

params0 = np.array([l1_prior_mode, l2_prior_mode])
param_bounds = [(0.1, 2), (0.001, 0.01)]

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
obs_std = 1e-3
obs_count_1 = 50
obs_count_2 = 50
obs_loc_1 = np.where(tt == 0.2)[0][0]
obs_loc_2 = np.where(tt == 0.8)[0][0]
obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_1+1))
obs_dict.update(util.sample_observations(uu, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
print("Number of observations:", obs_idxs.shape[0])

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dx, dt, params):
    """
    Constructs current linearised differential operator.
    """
    l1, l2 = params
    partial_t = FinDiff(1, dt, 1, acc=2)
    partial_x = FinDiff(0, dx, 1, acc=2)
    partial_xxx = FinDiff(0, dx, 3, acc=2)
    u0_x = partial_x(u0)
    diff_op = partial_t + Coef(l1 * u0) * partial_x + Coef(l1 * u0_x) * Identity() + Coef(l2) * partial_xxx
    return diff_op

def get_prior_mean(u0, diff_op_gen, params):
    """
    Calculates current prior mean.
    """
    l1, _ = params
    partial_x = FinDiff(0, dx, 1, acc=2)
    u0_x = partial_x(u0)
    diff_op = diff_op_gen(u0, params)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (l1 * u0 * u0_x).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u, params: get_diff_op(u, dx, dt, params)
prior_mean_gen = lambda u, params: get_prior_mean(u, diff_op_gen, params)

##################################
# log-pdf of parameter posterior #
##################################

def _logpdf_marginal_posterior(l1, l2, Q_u, Q_uy, Q_obs, mu_u, mu_uy, obs_dict, shape, regularisation=1e-3):
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

    # Compute l1 prior terms
    if l1 > 0:
        log_l1 = np.log(l1)
        log_p_l1 = np.log(tau_l1) - log_l1 - 0.5 * ((tau_l1 * (log_l1 - l1_0)) ** 2 + log_2pi) # log-normal prior
    else:
        log_p_l1 = float("-inf")

    # Compute l2 prior terms
    if l2 > 0:
        log_l2 = np.log(l2)
        log_p_l2 = np.log(tau_l2) - log_l2 - 0.5 * ((tau_l2 * (log_l2 - l2_0)) ** 2 + log_2pi) # log-normal prior
    else:
        log_p_l2 = float("-inf")

    # Compute GMRF prior terms
    diff_mu_uy_mu_u = mu_uy - mu_u
    log_p_ut = 0.5 * (Q_u_logdet - diff_mu_uy_mu_u.T @ Q_u @ diff_mu_uy_mu_u - M * log_2pi)

    # Compute obs model terms
    diff_obs_mu_uy = obs_vals - mu_uy[obs_idxs_flat]
    log_p_yut = 0.5 * (Q_obs_logdet - diff_obs_mu_uy.T @ Q_obs @ diff_obs_mu_uy - N * log_2pi)

    # Compute full conditional terms
    log_p_uyt = 0.5 * (Q_uy_logdet - M * log_2pi)

    logpdf = log_p_l1 + log_p_l2 + log_p_ut + log_p_yut - log_p_uyt
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
    res = linear._fit_gp(uu, obs_dict, obs_std, prior_mean, prior_precision, calc_std=return_conditional_params, calc_lml=False,
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


## Fit GP with non-linear SPDE prior from KdV equation

# Perform iterative optimisation
max_iter = 14
model = nonlinear.NonlinearINLASPDERegressor(uu, dx, dt, params0, diff_op_gen, prior_mean_gen, logpdf_marginal_posterior,
                                             mixing_coef=0.5, param_bounds=param_bounds, sampling_evec_scales=[0.0001, 0.0001],
                                             params_true=params_true)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True)
iter_count = len(model.mse_hist)

# Save animation
print("Saving animation...")
model.save_animation("figures/kdv/kdv_inla_iter_animation.gif", fps=3)
plt.show()

# Plot convergence history
plt.figure()
plt.plot(np.arange(1, iter_count + 1), model.mse_hist, label="Linearisation via expansion")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.xticks(np.arange(2, iter_count + 1, 2))
plt.legend()
plt.savefig("figures/kdv/mse_conv.png", dpi=200)
plt.show()

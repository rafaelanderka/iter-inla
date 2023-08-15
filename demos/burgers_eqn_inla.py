import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from scipy.special import loggamma
from findiff import FinDiff, Coef, Identity
from sksparse.cholmod import cholesky

from spdeinf import linear, metrics, nonlinear, plotting, util

# Define parameters of Burgers' equation
mu = 1 
nu_true = 0.1 # Kinematic viscosity coefficient

# Define parameters of the parameter priors
t_obs_a = 10.
t_obs_b = 0.1
tau_nu = 0.8
nu_0 = -1.
nu_prior_mode = np.exp(nu_0 - tau_nu ** (-2))

## Generate data from Burger's equation

# Create spatial discretisation
L_x = 10                      # Range of spatial domain
dx = 0.2                      # Spatial delta
N_x = int(L_x / dx) + 1       # Number of points in spatial discretisation
X = np.linspace(0, L_x, N_x)  # Spatial array

# Create temporal discretisation
L_t = 8                       # Range of temporal domain
dt = 0.2                      # Temporal delta
N_t = int(L_t / dt) + 1       # Number of points in temporal discretisation
T = np.linspace(0, L_t, N_t)  # Temporal array

# Create test points
Xgrid, Tgrid = np.meshgrid(T, X, indexing='ij')
X_test = np.stack([Tgrid.flatten(), Xgrid.flatten()], axis=1)

# Define wave number discretization
k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

# Define the initial condition    
# u0 = np.exp(-(X - 3)**2 / 2)
u0 = np.exp(-(X - 5)**2 / 10)

def burgers_odes(u, t, k, mu, nu):
    """
    Construct system of ODEs for Burgers' equation using method of lines
    Note we use a pseudo-spectral method s.t. we construct the (discrete) spatial derivatives in fourier space.
    PDE ---(FFT)---> ODE system
    """
    # Take x derivatives in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j * k * u_hat
    u_hat_xx = -k**2 * u_hat

    # Transform back to spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)

    # Assemble ODE
    u_t = -mu * u * u_x + nu * u_xx
    return u_t.real

# Solve system of ODEs
u = odeint(burgers_odes, u0, T, args=(k, mu, nu_true,), mxstep=5000).T

# Sample observations
obs_std = 1e-2
obs_count = 100
obs_dict = util.sample_observations(u, obs_count, obs_std, extent=(None, None, 0, None))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
print("Number of observations:", obs_idxs.shape[0])


################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dx, dt, nu):
    """
    Constructs current linearised differential operator.
    """
    partial_t = FinDiff(1, dt, 1)
    partial_x = FinDiff(0, dx, 1)
    partial_xx = FinDiff(0, dx, 2)
    u0_x = partial_x(u0)
    diff_op = partial_t + Coef(u0) * partial_x - Coef(nu) * partial_xx + Coef(u0_x) * Identity()
    return diff_op

def get_prior_mean(u0, nu, diff_op_gen):
    """
    Calculates current prior mean.
    """
    partial_x = FinDiff(0, dx, 1)
    u0_x = partial_x(u0)
    diff_op = diff_op_gen(u0, nu)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (u0 * u0_x).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u, nu: get_diff_op(u, dx, dt, nu)
prior_mean_gen = lambda u, nu: get_prior_mean(u, nu, diff_op_gen)


######################################
# Naive diff. op. and mean generator #
######################################

def get_diff_op_naive(u0, dx, dt, nu):
    partial_t = FinDiff(1, dt, 1)
    partial_x = FinDiff(0, dx, 1)
    partial_xx = FinDiff(0, dx, 2)
    diff_op = partial_t + Coef(u0) * partial_x - Coef(nu) * partial_xx
    return diff_op

def get_prior_mean_naive(u0, diff_op_gen):
    return np.zeros_like(u0)

diff_op_gen_naive = lambda u, nu: get_diff_op_naive(u, dx, dt, nu)
prior_mean_gen_naive = lambda u, nu: get_prior_mean_naive(u, diff_op_gen)


##################################
# log-pdf of parameter posterior #
##################################

def _logpdf_marginal_posterior(nu, t_obs, Q_u, Q_uy, Q_obs, mu_u, mu_uy, obs_dict, shape, regularisation=1e-5):
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

    # Compute nu prior terms
    if nu > 0:
        log_a = np.log(nu)
        log_p_a = np.log(tau_nu) - log_a - 0.5 * ((tau_nu * (log_a - nu_0)) ** 2 + log_2pi) # log-normal prior
    else:
        log_p_a = float("-inf")

    # Compute obs noise prior terms
    if t_obs >= 0:
        log_t_obs = np.log(t_obs)
        log_p_t_obs = (t_obs_a - 1) * log_t_obs - t_obs_b * t_obs + t_obs_a * np.log(t_obs_b) - loggamma(t_obs_a)
    else:
        log_p_t_obs = float("-inf")

    # Compute GMRF prior terms
    diff_mu_uy_mu_u = mu_uy - mu_u
    log_p_ua = 0.5 * (Q_u_logdet - diff_mu_uy_mu_u.T @ Q_u @ diff_mu_uy_mu_u - M * log_2pi)

    # Compute obs model terms
    diff_obs_mu_uy = obs_vals - mu_uy[obs_idxs_flat]
    log_p_yua = 0.5 * (Q_obs_logdet - diff_obs_mu_uy.T @ Q_obs @ diff_obs_mu_uy - N * log_2pi)

    # Compute full conditional terms
    log_p_uya = 0.5 * (Q_uy_logdet - M * log_2pi)

    logpdf = log_p_a + log_p_t_obs + log_p_ua + log_p_yua - log_p_uya
    # logpdf = log_p_ua
    # logpdf = log_p_a + log_p_t_obs
    return logpdf

def logpdf_marginal_posterior(x, u0, obs_dict, diff_op_gen, prior_mean_gen, return_conditional_params=False):
    # Unpack parameters
    nu = x[0]
    t_obs = x[1]
    obs_count = len(obs_dict.keys())

    # Compute prior mean
    prior_mean = prior_mean_gen(u0, nu)

    # Construct precision matrix corresponding to the linear differential operator
    diff_op_guess = diff_op_gen(u0, nu)
    L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
    LL = L.T @ L
    # LL_chol = cholesky(LL + tol * identity(LL.shape[0]))
    # kappa = LL_chol.spinv().diagonal().mean()
    # prior_precision = (self.sigma ** 2) / (self.dV * kappa) * LL
    # prior_precision = (self.sigma ** 2) / (self.dV) * LL
    prior_precision = LL

    # Get "data term" of full conditional
    res = linear._fit_gp(u, obs_dict, 1 / t_obs, prior_mean, prior_precision, calc_std=return_conditional_params, calc_lml=False,
                         include_initial_cond=False, return_posterior_precision=True, regularisation=1e-5)

    # Define prior and full condtional params
    mu_u = prior_mean
    Q_u = prior_precision
    mu_uy = res['posterior_mean']
    # mu_uy = u # For testing
    Q_uy = res['posterior_precision']
    Q_obs = sparse.diags([t_obs ** 2], 0, shape=(obs_count, obs_count), format='csc')

    # Compute marginal posterior
    logpdf = _logpdf_marginal_posterior(nu, t_obs, Q_u, Q_uy, Q_obs, mu_u.flatten(), mu_uy.flatten(), obs_dict, u0.shape)
    if return_conditional_params:
        return logpdf, mu_uy, res['posterior_var']
    return logpdf


## Fit GP with non-linear SPDE prior from Burgers' equation

# Perform iterative optimisation
max_iter = 10
model = nonlinear.NonlinearINLASPDERegressor(u, dx, dt, diff_op_gen, prior_mean_gen, logpdf_marginal_posterior, mixing_coef=1)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True)
iter_count = len(model.mse_hist)

# Save animation
print("Saving animation...")
model.save_animation("figures/burgers_eqn/burgers_eqn_inla_iter_animation.gif", fps=10)

# # Check prior covariance
# diff_op_init = diff_op_gen(np.zeros_like(u))
# diff_op_final = diff_op_gen(model.posterior_mean)

# L_init = util.operator_to_matrix(diff_op_init, u.shape, interior_only=False)
# L_final = util.operator_to_matrix(diff_op_final, u.shape, interior_only=False)
# LL_init = L_init.T @ L_init
# LL_final = L_final.T @ L_final
# LL_init_chol = cholesky(LL_init + identity(LL_init.shape[0]))
# LL_final_chol = cholesky(LL_final + identity(LL_final.shape[0]))
# prior_init_std = np.sqrt(LL_init_chol.spinv().diagonal().reshape(u.shape))
# prior_final_std = np.sqrt(LL_final_chol.spinv().diagonal().reshape(u.shape))

# fig, ax = plt.subplots(1, 2)
# im_init = ax[0].imshow(prior_init_std, origin="lower")
# im_final = ax[1].imshow(prior_final_std, origin="lower")
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('t')
# fig.colorbar(im_init)
# fig.colorbar(im_final)
# plt.show()

# # Fit with naive linearisation
# model_naive = nonlinear.NonlinearINLASPDERegressor(u, dx, dt, diff_op_gen_naive, prior_mean_gen_naive)
# model_naive.fit(obs_dict, obs_std, max_iter=max_iter, animated=False, calc_std=True)
# iter_count_naive = len(model_naive.mse_hist)

# # Plot convergence history
# plt.plot(np.arange(1, iter_count + 1), model.mse_hist, label="Linearisation via expansion")
# plt.plot(np.arange(1, iter_count_naive + 1), model_naive.mse_hist, label="Naive linearisation")
# plt.yscale('log')
# plt.xlabel("Iteration")
# plt.ylabel("MSE")
# plt.xticks(np.arange(2, max(iter_count, iter_count_naive) + 1, 2))
# plt.legend()
# plt.savefig("figures/burgers_eqn/inla_mse_conv.png", dpi=200)
# plt.show()

# Fit with RBF
# posterior_mean_rbf, posterior_std_rbf = linear.fit_rbf_gp(u, obs_dict, X_test, dx, dt, obs_std)
# print(metrics.mse(u, posterior_mean_rbf))

# Plot results
# plot_kwargs = {
#         'mean_vmin': u.min(),
#         'mean_vmax': u.max(),
#         'std_vmin': 0,
#         'std_vmax': model.posterior_std.max(),
#         'diff_vmin': -0.2,
#         'diff_vmax': 1,
#         }
# plotting.plot_gp_2d(u, model.posterior_mean, model.posterior_std, obs_idxs, 'figures/burgers_eqn/burgers_eqn_20_iter.png', **plot_kwargs)
# plotting.plot_gp_2d(u, posterior_mean_rbf, posterior_std_rbf, obs_idxs, 'figures/burgers_eqn/burgers_eqn_rbf.png', **plot_kwargs)

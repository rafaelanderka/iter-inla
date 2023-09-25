import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from sksparse.cholmod import cholesky
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
from findiff import FinDiff, Coef, Identity

from spdeinf import linear, metrics, nonlinear, plotting, util

# Set seed
np.random.seed(1)

# Define damped pendulum eqn. and observation parameters
b = 0.2
c = 1.
obs_std = 1e-1
params_true = np.array([b, c, 1 / obs_std])
print("True parameters:", params_true)

# Define parameters of the parameter priors
tau_b = 1
b_prior_mode = 0.2
b_0 = np.log(b_prior_mode) + (tau_b ** (-2))

tau_c = 1
c_prior_mode = 1.
c_0 = np.log(c_prior_mode) + (tau_c ** (-2))

tau_t_obs = 0.1
t_obs_prior_mode = 10
t_obs_0 = np.log(t_obs_prior_mode) + (tau_t_obs ** (-2))

params0 = np.array([b_prior_mode, c_prior_mode, t_obs_prior_mode])
param_bounds = [(0.1, 1), (0.1, 15), (1, 100)]
print("Prior parameters (mode):", params0)

# Define fitting hyperparameters
max_iter = 20

# Visualise priors
b_linspace = np.linspace(0, 10, 1000)
c_linspace = np.linspace(0, 100, 1000)
t_obs_linspace = np.linspace(0, 1000, 1000)
def lognormal_pdf(x, x_0, tau_x):
    log_x = np.log(x)
    log_2pi = np.log(2 * np.pi)
    log_p_x = np.log(tau_x) - log_x - 0.5 * ((tau_x * (log_x - x_0)) ** 2 + log_2pi)
    return np.exp(log_p_x)
fig, ax = plt.subplots(3, 1, figsize=(10,10))
ax[0].plot(b_linspace, lognormal_pdf(b_linspace, b_0, tau_b))
ax[0].set_xlabel('b')
ax[1].plot(c_linspace, lognormal_pdf(c_linspace, c_0, tau_c))
ax[1].set_xlabel('c')
ax[2].plot(t_obs_linspace, lognormal_pdf(t_obs_linspace, t_obs_0, tau_t_obs))
ax[2].set_xlabel('t_obs')
plt.grid(True)
plt.show()

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dt, params):
    """
    Constructs current linearised differential operator.
    """
    b, c, _ = params
    partial_t = FinDiff(1, dt, 1)
    partial_tt = FinDiff(1, dt, 2)
    u0_cos = np.cos(u0)
    diff_op = partial_tt + Coef(b) * partial_t + Coef(c * u0_cos) * Identity()
    return diff_op

def get_prior_mean(u0, diff_op_gen, params):
    """
    Calculates current prior mean.
    """
    b, c, _ = params
    u0_cos = np.cos(u0)
    u0_sin = np.sin(u0)
    diff_op = diff_op_gen(u0, params)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (c * (u0 * u0_cos - u0_sin)).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u, params: get_diff_op(u, dt, params)
prior_mean_gen = lambda u, params: get_prior_mean(u, diff_op_gen, params)
diff_op_gen_known = lambda u: get_diff_op(u, dt, (b, c, None))
prior_mean_gen_known = lambda u: get_prior_mean(u, diff_op_gen, (b, c, None))

##################################
# log-pdf of parameter posterior #
##################################

def _logpdf_marginal_posterior(b, c, t_obs, Q_u, Q_uy, Q_obs, mu_u, mu_uy, obs_dict, shape, regularisation=1e-3):
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

    # # Compute obs noise prior terms
    if t_obs >= 0:
        log_t_obs = np.log(t_obs)
        log_p_t_obs = np.log(tau_t_obs) - log_t_obs - 0.5 * ((tau_t_obs * (log_t_obs - t_obs_0)) ** 2 + log_2pi) # log-normal prior
    else:
        log_p_t_obs = float("-inf")

    # Compute GMRF prior terms
    diff_mu_uy_mu_u = mu_uy - mu_u
    log_p_ut = 0.5 * (Q_u_logdet - diff_mu_uy_mu_u.T @ Q_u @ diff_mu_uy_mu_u - M * log_2pi)

    # Compute obs model terms
    diff_obs_mu_uy = obs_vals - mu_uy[obs_idxs_flat]
    log_p_yut = 0.5 * (Q_obs_logdet - diff_obs_mu_uy.T @ Q_obs @ diff_obs_mu_uy - N * log_2pi)

    # Compute full conditional terms
    log_p_uyt = 0.5 * (Q_uy_logdet - M * log_2pi)

    # print(log_p_b, log_p_c, log_p_ut, log_p_yut, log_p_uyt)
    logpdf = log_p_b + log_p_c + log_p_t_obs + log_p_ut + log_p_yut - log_p_uyt
    # logpdf = log_p_b + log_p_c
    # logpdf = log_p_ut
    return logpdf

def logpdf_marginal_posterior(x, u0, obs_dict, diff_op_gen, prior_mean_gen, return_conditional_params=False, debug=False):
    # Process args
    obs_count = len(obs_dict.keys())
    t_obs = x[2]
    # t_obs = 10

    # Compute prior mean
    prior_mean = prior_mean_gen(u0, x)

    # Construct precision matrix corresponding to the linear differential operator
    diff_op_guess = diff_op_gen(u0, x)
    L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
    prior_precision = L.T @ L

    # Get "data term" of full conditional
    res = linear._fit_gmrf(u, obs_dict, 1 / t_obs, prior_mean, prior_precision, calc_std=return_conditional_params,
                         include_initial_cond=False, return_posterior_precision=True, regularisation=1e-5)

    # Define prior and full condtional params
    mu_u = prior_mean
    Q_u = prior_precision
    mu_uy = res['posterior_mean']
    Q_uy = res['posterior_precision']
    Q_obs = sparse.diags([t_obs ** 2], 0, shape=(obs_count, obs_count), format='csc')

    if debug:
        plt.figure()
        plt.plot(prior_mean[0])
        plt.show()

    # Compute marginal posterior
    logpdf = _logpdf_marginal_posterior(x[0], x[1], x[2], Q_u, Q_uy, Q_obs, mu_u.flatten(), mu_uy.flatten(), obs_dict, u0.shape)
    if return_conditional_params:
        return logpdf, mu_uy, res['posterior_var']
    return logpdf


################################
#      Dataset Generation      #
################################

## Generate datasets
# Create temporal discretisation
L_t = 24.3                    # Duration of simulation [s]
dt = 0.1                      # Infinitesimal time
N_t = int(L_t / dt) + 1       # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array
T = np.around(T, decimals=1)
shape = (1, N_t)

# Define the initial condition    
u0 = [0.75 * np.pi, 0.]

# Define corresponding system of ODEs
def pend(u, t, b, c):
    theta, omega = u
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

# Solve system of ODEs
u_full = odeint(pend, u0, T, args=(b, c,))

# For our purposes we only need the solution for the pendulum angle, theta
u = u_full[:, 0].reshape(1, -1)

# Sample 16 observations in [0, 6]
obs_count = 16
obs_loc_1 = np.where(T == 6.)[0][0]
test_start_idx = obs_loc_1 + 1
datasets = []
num_repeats = 2
print("Generating datasets...")
for i in range(num_repeats):
    obs_dict = util.sample_observations(u, obs_count, obs_std, extent=(None, None, 0, obs_loc_1+1))
    datasets.append(obs_dict)

    obs_table = np.empty((obs_count, 2))
    obs_table = [[T[k[1]], v] for k, v in obs_dict.items()]
    util.obs_to_csv(obs_table, header=["t", "theta"], filename=f"data/PendulumTrain{i}.csv")

    u_table = np.empty((N_t, 2))
    u_table[:,0] = T.flatten()
    u_table[:,1] = u.flatten()
    util.obs_to_csv(u_table, header=["t", "theta"], filename=f"data/PendulumTest{i}.csv")


################################
#          Benchmarks          #
################################

## Standard GP regression
# Define the GP kernel: RBF + WhiteKernel (to model the noise)
gp_kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=obs_std**2, noise_level_bounds="fixed")

# Fit GP
print("===== GPR (RBF Cov.) =====")
gp_rmses = np.empty(num_repeats)
gp_mnlls = np.empty(num_repeats)
ic_means = np.empty((num_repeats, 2))
ic_covs = np.empty((num_repeats, 2, 2))
ic_stds = np.empty((num_repeats, 2))
fig, axs = plotting.init_gp_1d_plot(num_repeats)
for i, obs_dict in enumerate(datasets):
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    obs_vals = np.array(list(obs_dict.values()), dtype=float)
    obs_locs = np.array([[0, T[j]] for i, j in obs_idxs])
    gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=10)
    gp.fit(obs_locs, obs_vals)
    pred_locs = [[0, t] for t in T]
    gp_mean, gp_std = gp.predict(pred_locs, return_std=True)
    gp_mean = gp_mean.reshape(1,-1)
    gp_std = gp_std.reshape(1,-1)
    gp_rmse = metrics.rmse(gp_mean.squeeze()[test_start_idx:], u.squeeze()[test_start_idx:])
    gp_rmses[i] = gp_rmse
    gp_mnll = -stats.norm.logpdf(u.flatten()[test_start_idx:], loc=gp_mean.flatten()[test_start_idx:], scale=gp_std.flatten()[test_start_idx:]).mean()
    gp_mnlls[i] = gp_mnll
    print(f"i={i}, RMSE={gp_rmse}, MNLL={gp_mnll}")
    plotting.add_gp_1d_plot(fig, axs, i, u, gp_mean, gp_std, obs_idxs, obs_vals)

    # Get initial condition for other benchmarks
    test_locs_t0 = [[0, 0], [0, dt]]
    ic_means[i], ic_covs[i] = gp.predict(test_locs_t0, return_cov=True)
    ic_stds[i] = np.sqrt(np.diag(ic_covs[i]))

print(f"RMSE = {np.mean(gp_rmses)} +- {np.std(gp_rmses)}")
print(f"MNLL = {np.mean(gp_mnlls)} +- {np.std(gp_mnlls)}")
fig.tight_layout()
plt.show()


# Plot fit
plt.figure(figsize=(3,3))
plt.plot(T, u.squeeze(), "b", label="Ground truth")
plt.plot(T, gp_mean.squeeze(), "k", label="Posterior mean")
plt.plot(T, gp_mean.squeeze() + gp_std.squeeze(), "--", color="grey", label="Posterior std. dev.")
plt.plot(T, gp_mean.squeeze() - gp_std.squeeze(), "--", color="grey")
plt.axvline(dt * obs_loc_1, color='grey', ls=':')
plt.scatter(dt * obs_idxs[:,1], obs_vals, c="r", marker="x", label="Observations")
plt.legend(loc="upper right")
plt.xlabel("$t$")
plt.ylabel("$u$", labelpad=0)
plt.tight_layout()
plt.savefig("figures/pendulum/pendulum_gp.pdf")
plt.show()

# Get IC/background from GPR fit
ic_idxs = np.zeros((1, 2), dtype=int)
ic_idxs[:,0] = np.arange(1, dtype=int)

# Save data for other benchmarks
for i in range(num_repeats):
    data_dict = {'uu': u, 'uu_full': u_full, 'tt': T, 'dt': dt, 'u0_mean': ic_means[i], 'u0_cov': ic_covs[i], 'u0_std': ic_stds[i], 'obs_dict': datasets[i], 'obs_std': obs_std}
    data_file = f"data/pendulum_{i}.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

## SPDE GMRF regression with known parameters
# Perform iterative optimisation
print("===== SPDE GMRF (known params) =====")
spde_rmses = np.empty(num_repeats)
spde_mnlls = np.empty(num_repeats)
fig, axs = plotting.init_gp_1d_plot(num_repeats)
for i, obs_dict in enumerate(datasets):
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    obs_vals = np.array(list(obs_dict.values()), dtype=float)
    obs_locs = np.array([[0, T[j]] for i, j in obs_idxs])
    model = nonlinear.NonlinearSPDERegressor(u, 1, dt, diff_op_gen_known, prior_mean_gen_known, mixing_coef=1.)
    spde_u0, spde_mean, spde_std = model.fit(obs_dict, obs_std, max_iter=max_iter, animated=False, calc_std=True, calc_mnll=True)
    spde_rmse = metrics.rmse(spde_mean.squeeze()[test_start_idx:], u.squeeze()[test_start_idx:])
    spde_rmses[i] = spde_rmse
    spde_mnll = -stats.norm.logpdf(u.flatten()[test_start_idx:], loc=spde_mean.flatten()[test_start_idx:], scale=spde_std.flatten()[test_start_idx:]).mean()
    spde_mnlls[i] = spde_mnll
    print(f"i={i}, RMSE={spde_rmse}, MNLL={spde_mnll}")
    plotting.add_gp_1d_plot(fig, axs, i, u, spde_mean, spde_std, obs_idxs, obs_vals)
print(f"RMSE = {np.mean(spde_rmses)} +- {np.std(spde_rmses)}")
print(f"MNLL = {np.mean(spde_mnlls)} +- {np.std(spde_mnlls)}")
fig.tight_layout()
plt.show()

# Plot fit
plt.figure(figsize=(3,3))
plt.plot(T, u.squeeze(), "b", label="Ground truth")
plt.plot(T, spde_mean.squeeze(), "k", label="Posterior mean")
plt.plot(T, spde_mean.squeeze() + spde_std.squeeze(), "--", color="grey", label="Posterior std. dev.")
plt.plot(T, spde_mean.squeeze() - spde_std.squeeze(), "--", color="grey")
plt.axvline(dt * obs_loc_1, color='grey', ls=':')
plt.scatter(dt * obs_idxs[:,1], obs_vals, c="r", marker="x", label="Observations")
plt.xlabel("$t$")
plt.ylabel("$u$", labelpad=0)
plt.tight_layout()
plt.savefig("figures/pendulum/pendulum_spde.pdf")
plt.show()

## SPDE GMRF regression with unknown parameters
# Perform iterative optimisation
print("===== SPDE GMRF (unknown params) =====")
spde_rmses = np.empty(num_repeats)
spde_mnlls = np.empty(num_repeats)
fig, axs = plotting.init_gp_1d_plot(num_repeats)
for i, obs_dict in enumerate(datasets):
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    obs_vals = np.array(list(obs_dict.values()), dtype=float)
    obs_locs = np.array([[0, T[j]] for i, j in obs_idxs])

    model = nonlinear.NonlinearINLASPDERegressor(u, 1, dt, params0, diff_op_gen, prior_mean_gen, logpdf_marginal_posterior,
                                                 mixing_coef=0.5, param_bounds=param_bounds, sampling_evec_scales=[3e-1, 1e-1, 1e-2],
                                                 sampling_threshold=1, params_true=params_true)
    spde_u0, spde_mean, spde_std = model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)
    spde_rmse = metrics.rmse(spde_u0.squeeze()[test_start_idx:], u.squeeze()[test_start_idx:])
    spde_rmses[i] = spde_rmse
    spde_mnll = -stats.norm.logpdf(u.flatten()[test_start_idx:], loc=spde_u0.flatten()[test_start_idx:], scale=spde_std.flatten()[test_start_idx:]).mean()
    spde_mnlls[i] = spde_mnll
    print(f"i={i}, RMSE={spde_rmse}, MNLL={spde_mnll}")
    plotting.add_gp_1d_plot(fig, axs, i, u, spde_mean, spde_std, obs_idxs, obs_vals)
print(f"RMSE = {np.mean(spde_rmses)} +- {np.std(spde_rmses)}")
print(f"MNLL = {np.mean(spde_mnlls)} +- {np.std(spde_mnlls)}")
fig.tight_layout()
plt.show()

# Plot fit
plt.figure(figsize=(3,3))
plt.plot(T, u.squeeze(), "b", label="Ground truth")
plt.plot(T, spde_mean.squeeze(), "k", label="Posterior mean")
plt.plot(T, spde_mean.squeeze() + spde_std.squeeze(), "--", color="grey", label="Posterior std. dev.")
plt.plot(T, spde_mean.squeeze() - spde_std.squeeze(), "--", color="grey")
plt.axvline(dt * obs_loc_1, color='grey', ls=':')
plt.scatter(dt * obs_idxs[:,1], obs_vals, c="r", marker="x", label="Observations")
plt.xlabel("$t$")
plt.ylabel("$u$", labelpad=0)
plt.tight_layout()
plt.savefig("figures/pendulum/pendulum_spde_inla.pdf")
plt.show()

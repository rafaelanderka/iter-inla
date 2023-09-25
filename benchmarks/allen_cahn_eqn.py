import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import sparse, stats
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from findiff import FinDiff, Coef, Identity

from spdeinf import linear, metrics, nonlinear, plotting, util

# Set seed
np.random.seed(1)

# Define Allen-Cahn eqn. and observation parameters
alpha = 0.0001
beta = 5
obs_std = 1e-2
params_true = np.array([alpha, beta, 1 / obs_std])
print("True parameters:", params_true)

# Define parameters of the parameter priors
tau_alpha = 0.1
alpha_prior_mode = 0.0001
alpha_0 = np.log(alpha_prior_mode) + (tau_alpha ** (-2))

tau_beta = 0.1
beta_prior_mode = 5.
beta_0 = np.log(beta_prior_mode) + (tau_beta ** (-2))

tau_t_obs = 0.1
t_obs_prior_mode = 100
t_obs_0 = np.log(t_obs_prior_mode) + (tau_t_obs ** (-2))
params0 = np.array([alpha_prior_mode, beta_prior_mode, t_obs_prior_mode])
param_bounds = [(0.00005, 0.005), (1, 10), (50, 200)]
print("Prior parameters (mode):", params0)

# Define fitting hyperparameters
max_iter = 20

# Visualise priors
alpha_linspace = np.linspace(0, 100, 1000)
beta_linspace = np.linspace(0, 0.1, 1000)
t_obs_linspace = np.linspace(0, 1000, 1000)
def lognormal_pdf(x, x_0, tau_x):
    log_x = np.log(x)
    log_2pi = np.log(2 * np.pi)
    log_p_x = np.log(tau_x) - log_x - 0.5 * ((tau_x * (log_x - x_0)) ** 2 + log_2pi)
    return np.exp(log_p_x)
fig, ax = plt.subplots(3, 1, figsize=(10,10))
ax[0].plot(alpha_linspace, lognormal_pdf(alpha_linspace, alpha_0, tau_alpha))
ax[0].set_xlabel('alpha')
ax[1].plot(beta_linspace, lognormal_pdf(beta_linspace, beta_0, tau_beta))
ax[1].set_xlabel('beta')
ax[2].plot(t_obs_linspace, lognormal_pdf(t_obs_linspace, t_obs_0, tau_t_obs))
ax[2].set_xlabel('t_obs')
plt.grid(True)
plt.show()

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dx, dt, params):
    """
    Constructs current linearised differential operator.
    """
    alpha, beta, _ = params
    partial_t = FinDiff(1, dt, 1, acc=4)
    partial_xx = FinDiff(0, dx, 2, acc=2, periodic=True)
    u0_sq = u0 ** 2
    diff_op = partial_t - Coef(alpha) * partial_xx + Coef(3 * beta * u0_sq) * Identity() - Coef(beta) * Identity()
    return diff_op

def get_prior_mean(u0, diff_op_gen, params):
    """
    Calculates current prior mean.
    """
    beta = params[1]
    u0_cu = u0 ** 3
    diff_op = diff_op_gen(u0, params)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (2 * beta * u0_cu).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u, params: get_diff_op(u, dx, dt, params)
prior_mean_gen = lambda u, params: get_prior_mean(u, diff_op_gen, params)
diff_op_gen_known = lambda u: get_diff_op(u, dx, dt, (alpha, beta, None))
prior_mean_gen_known = lambda u: get_prior_mean(u, diff_op_gen, (alpha, beta, None))

##################################
# log-pdf of parameter posterior #
##################################

def _logpdf_marginal_posterior(alpha, beta, t_obs, Q_u, Q_uy, Q_obs, mu_u, mu_uy, obs_dict, shape, regularisation=1e-3):
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

    # Compute alpha prior terms
    if alpha > 0:
        log_alpha = np.log(alpha)
        log_p_alpha = np.log(tau_alpha) - log_alpha - 0.5 * ((tau_alpha * (log_alpha - alpha_0)) ** 2 + log_2pi) # log-normal prior
    else:
        log_p_alpha = float("-inf")

    # Compute beta prior terms
    if beta > 0:
        log_beta = np.log(beta)
        log_p_beta = np.log(tau_beta) - log_beta - 0.5 * ((tau_beta * (log_beta - beta_0)) ** 2 + log_2pi) # log-normal prior
    else:
        log_p_beta = float("-inf")

    # Compute t_obs prior terms
    if t_obs > 0:
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

    logpdf = log_p_alpha + log_p_beta + log_p_t_obs + log_p_ut + log_p_yut - log_p_uyt
    return logpdf

def logpdf_marginal_posterior(x, u0, obs_dict, diff_op_gen, prior_mean_gen, return_conditional_params=False, debug=False):
    # Process args
    obs_count = len(obs_dict.keys())
    t_obs = x[2]

    # Compute prior mean
    prior_mean = prior_mean_gen(u0, x)

    # Construct precision matrix corresponding to the linear differential operator
    diff_op_guess = diff_op_gen(u0, x)
    L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
    prior_precision = L.T @ L

    # Get "data term" of full conditional
    res = linear._fit_gmrf(uu, obs_dict, 1 / t_obs, prior_mean, prior_precision, calc_std=return_conditional_params,
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

# Sample 256 observations in [0, 2.8]
obs_count_1 = 256
obs_loc_1 = np.where(tt == 0.0)[0][0]
obs_loc_2 = np.where(tt == 0.28)[0][0]
datasets = []
num_repeats = 5
print(uu.shape)
print("Generating datasets...")
for i in range(num_repeats):
    obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_2+1))
    datasets.append(obs_dict)

    obs_table = np.empty((obs_count_1, 3))
    obs_table = [[xx[k[0]], tt[k[1]], v] for k, v in obs_dict.items()]
    util.obs_to_csv(obs_table, header="XTU", filename=f"data/ACTrain{i}.csv")

    u_table = np.empty((N_x * N_t, 3))
    u_table[:,0] = Xgrid.flatten()
    u_table[:,1] = Tgrid.flatten()
    u_table[:,2] = uu.flatten()
    util.obs_to_csv(u_table, header="XTU", filename=f"data/ACTest{i}.csv")


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
ic_means = np.empty((num_repeats, N_x))
ic_covs = np.empty((num_repeats, N_x, N_x))
ic_stds = np.empty((num_repeats, N_x))
fig, axs = plotting.init_gp_2d_plot(num_repeats)
for i, obs_dict in enumerate(datasets):
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    obs_vals = np.array(list(obs_dict.values()), dtype=float)
    obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
    gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=10)
    gp.fit(obs_locs, obs_vals)
    test_locs = [[x, t] for x in xx for t in tt]
    gp_mean, gp_std = gp.predict(test_locs, return_std=True)
    gp_mean = gp_mean.reshape(shape)
    gp_std = gp_std.reshape(shape)
    gp_rmse = metrics.rmse(gp_mean, uu)
    gp_rmses[i] = gp_rmse
    gp_mnll = -stats.norm.logpdf(uu.flatten(), loc=gp_mean.flatten(), scale=gp_std.flatten()).mean()
    gp_mnlls[i] = gp_mnll
    ic_means[i] = gp_mean[:,0]
    ic_stds[i] = gp_std[:,0]
    print(f"i={i}, RMSE={gp_rmse}, MNLL={gp_mnll}")
    plotting.add_gp_2d_plot(fig, axs, i, uu, gp_mean, gp_std, obs_idxs)

    # Get initial condition for other benchmarks
    test_locs_t0 = [[x, 0] for x in xx]
    ic_means[i], ic_covs[i] = gp.predict(test_locs_t0, return_cov=True)
    ic_stds[i] = np.sqrt(np.diag(ic_covs[i]))

print(f"RMSE = {np.mean(gp_rmses)} +- {np.std(gp_rmses)}")
print(f"MNLL = {np.mean(gp_mnlls)} +- {np.std(gp_mnlls)}")
fig.tight_layout()
plt.show()

# Plot fit
plt.figure(figsize=(3,3))
im = plt.imshow(uu, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
obs_handle = plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_gt.pdf", transparent=True)

plt.figure(figsize=(3,3))
plt.imshow(gp_mean, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_gp.pdf", transparent=True)

plt.figure(figsize=(3,3))
im_std = plt.imshow(gp_std, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r", norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=3))
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_gp_std.pdf", transparent=True)

fig, ax = plt.subplots(5, 2, figsize=(3,3), width_ratios=[0.0, 1.0], height_ratios=[0.35, 0.10, 0.225, 0.10, 0.225])
ax = ax.flatten()
for i in range(len(ax)):
    if i == 1 or i == 3 or i == 7: continue
    ax[i].remove()
ax[1].legend(loc="center", handles=[obs_handle])
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].get_xaxis().set_ticks([])
ax[1].get_yaxis().set_ticks([])
plt.colorbar(im, cax=ax[3], location="bottom")
cb_std = plt.colorbar(im_std, cax=ax[7], location="bottom")
minorticks = np.concatenate([np.arange(1, 10)/100000., np.arange(1, 10)/10000., np.arange(1, 10)/1000., np.arange(1, 10)/100., np.arange(1, 10)/10., np.arange(1, 4)])
cb_std.ax.xaxis.set_ticks(minorticks, minor=True)
ax[3].set_xlabel("Mean")
ax[7].set_xlabel("Standard deviation")
plt.savefig("figures/allen_cahn_eqn/ac_colorbars.pdf", transparent=True)
plt.show()

# Get IC/background from GPR fit
ic_idxs = np.zeros((N_x, 2), dtype=int)
ic_idxs[:,0] = np.arange(N_x, dtype=int)

# Save data for other benchmarks
for i in range(num_repeats):
    data_dict = {'uu': uu, 'uu_full': uu_full, 'xx': xx, 'tt': tt, 'tt_full': tt_full, 'dx': dx, 'dt': dt, 'dx_full': dx_full, 'dt_full': dt_full, 'u0_mean': ic_means[i], 'u0_cov': ic_covs[i], 'u0_std': ic_stds[i], 'obs_dict': datasets[i], 'obs_std': obs_std}
    data_file = f"data/ac_{i}.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

## SPDE GMRF regression with known parameters
# Perform iterative optimisation
print("===== SPDE GMRF (known params) =====")
spde_rmses = np.empty(num_repeats)
spde_mnlls = np.empty(num_repeats)
fig, axs = plotting.init_gp_2d_plot(num_repeats)
for i, obs_dict in enumerate(datasets):
    obs_dict_ext = obs_dict.copy()
    obs_dict_ext.update({tuple(key): value for key, value, std in zip(ic_idxs, ic_means[i], ic_stds[i])})
    model = nonlinear.NonlinearSPDERegressor(uu, dx, dt, diff_op_gen_known, prior_mean_gen_known, mixing_coef=1.)
    spde_u0, spde_mean, spde_std = model.fit(obs_dict_ext, obs_std, max_iter=max_iter, animated=False, calc_std=True, calc_mnll=True)
    spde_rmse = metrics.rmse(spde_mean, uu)
    spde_rmses[i] = spde_rmse
    spde_mnll = -stats.norm.logpdf(uu.flatten(), loc=spde_mean.flatten(), scale=spde_std.flatten()).mean()
    spde_mnlls[i] = spde_mnll
    print(f"i={i}, RMSE={spde_rmse}, MNLL={spde_mnll}")
    obs_idxs = np.array(list(obs_dict_ext.keys()), dtype=int)
    plotting.add_gp_2d_plot(fig, axs, i, uu, spde_mean, spde_std, obs_idxs)
print(f"RMSE = {np.mean(spde_rmses)} +- {np.std(spde_rmses)}")
print(f"MNLL = {np.mean(spde_mnlls)} +- {np.std(spde_mnlls)}")
fig.tight_layout()
plt.show()

plt.figure(figsize=(3,3))
plt.imshow(spde_mean, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_spde.pdf", transparent=True)

plt.figure(figsize=(3,3))
plt.imshow(spde_std, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r", norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=3))
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_spde_std.pdf", transparent=True)
plt.show()

## SPDE GMRF regression with unknown parameters
# Perform iterative optimisation
print("===== SPDE GMRF (unknown params) =====")
spde_rmses = np.empty(num_repeats)
spde_mnlls = np.empty(num_repeats)
fig, axs = plotting.init_gp_2d_plot(num_repeats)
for i, obs_dict in enumerate(datasets):
    obs_dict_ext = obs_dict.copy()
    obs_dict_ext.update({tuple(key): value for key, value, std in zip(ic_idxs, ic_means[i], ic_stds[i])})
    model = nonlinear.NonlinearINLASPDERegressor(uu, dx, dt, params0, diff_op_gen, prior_mean_gen, logpdf_marginal_posterior,
                                                 mixing_coef=0.5, param_bounds=param_bounds, sampling_evec_scales=[1, 1, 1],
                                                 sampling_threshold=1, params_true=params_true)
    spde_u0, spde_mean, spde_std = model.fit(obs_dict_ext, obs_std, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)
    spde_rmse = metrics.rmse(spde_mean, uu)
    spde_rmses[i] = spde_rmse
    spde_mnll = -stats.norm.logpdf(uu.flatten(), loc=spde_mean.flatten(), scale=spde_std.flatten()).mean()
    spde_mnlls[i] = spde_mnll
    print(f"i={i}, RMSE={spde_rmse}, MNLL={spde_mnll}")
    obs_idxs = np.array(list(obs_dict_ext.keys()), dtype=int)
    plotting.add_gp_2d_plot(fig, axs, i, uu, spde_mean, spde_std, obs_idxs)
print(f"RMSE = {np.mean(spde_rmses)} +- {np.std(spde_rmses)}")
print(f"MNLL = {np.mean(spde_mnlls)} +- {np.std(spde_mnlls)}")
fig.tight_layout()
plt.show()

plt.figure(figsize=(3,3))
plt.imshow(spde_mean, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_spde_inla.pdf", transparent=True)

plt.figure(figsize=(3,3))
plt.imshow(spde_std, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r", norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=3))
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=0.6, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/allen_cahn_eqn/ac_spde_inla_std.pdf", transparent=True)
plt.show()

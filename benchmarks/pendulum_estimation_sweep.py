import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from sksparse.cholmod import cholesky
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from findiff import FinDiff, Coef, Identity

from spdeinf import linear, metrics, nonlinear, plotting, util

# Set seed
np.random.seed(1)

# Define damped pendulum eqn. and observation parameters
b = 0.2
c = 1.
obs_std = 1e-2
params_true = np.array([b, c])
print("True parameters:", params_true)
param_bounds = [(0.01, 1), (0.1, 5)]

# Define fitting hyperparameters
max_iter = 3

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
diff_op_gen_known = lambda u: get_diff_op(u, dt, (b, c, None))
prior_mean_gen_known = lambda u: get_prior_mean(u, diff_op_gen, (b, c, None))

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

    # Compute GMRF prior terms
    diff_mu_uy_mu_u = mu_uy - mu_u
    log_p_ut = 0.5 * (Q_u_logdet - diff_mu_uy_mu_u.T @ Q_u @ diff_mu_uy_mu_u - M * log_2pi)

    # Compute obs model terms
    diff_obs_mu_uy = obs_vals - mu_uy[obs_idxs_flat]
    log_p_yut = 0.5 * (Q_obs_logdet - diff_obs_mu_uy.T @ Q_obs @ diff_obs_mu_uy - N * log_2pi)

    # Compute full conditional terms
    log_p_uyt = 0.5 * (Q_uy_logdet - M * log_2pi)

    # We assume uniform parameter prior
    logpdf = log_p_ut + log_p_yut - log_p_uyt
    return logpdf

def logpdf_marginal_posterior(x, u0, obs_dict, diff_op_gen, prior_mean_gen, return_conditional_params=False, debug=False):
    # Process args
    obs_count = len(obs_dict.keys())

    # Compute prior mean
    prior_mean = prior_mean_gen(u0, x)

    # Construct precision matrix corresponding to the linear differential operator
    diff_op_guess = diff_op_gen(u0, x)
    L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
    prior_precision = L.T @ L

    # Get "data term" of full conditional
    res = linear._fit_gmrf(u, obs_dict, obs_std, prior_mean, prior_precision, calc_std=return_conditional_params,
                         include_initial_cond=False, return_posterior_precision=True, regularisation=1e-5)

    # Define prior and full condtional params
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
    if return_conditional_params:
        return logpdf, mu_uy, res['posterior_var']
    return logpdf


################################
#      Dataset Generation      #
################################

## Generate datasets
# Create temporal discretisation
L_t = 24.3                    # Duration of simulation [s]
dt = 0.01                     # Infinitesimal time
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

# Main sweep
min_window = 10
max_window = 2430
num_repeats = 1
obs_spacing = round(0.1 / dt) # Ensure equal obs. regardless of time step
windows = np.arange(min_window, max_window + 1, obs_spacing)
params_opt = np.zeros((len(windows), num_repeats, 2))
rmses = np.zeros((len(windows), num_repeats))
mnlls = np.zeros((len(windows), num_repeats))
for k_idx, k in enumerate(windows):
    obs_count = k
    obs_loc_1 = k
    test_start_idx = obs_loc_1 + 1
    datasets = []
    for i in range(num_repeats):
        obs_idxs = np.arange(0, k, obs_spacing)
        obs_val = u[0,obs_idxs]
        obs_dict = {(0, idx): val for idx, val in zip(obs_idxs, obs_val)}
        datasets.append(obs_dict)

        obs_table = np.empty((obs_count, 2))
        obs_table = [[T[k[1]], v] for k, v in obs_dict.items()]
        util.obs_to_csv(obs_table, header=["t", "theta"], filename=f"data/PendulumParamSweepTrain_{k}_{i}.csv")

        u_table = np.empty((N_t, 2))
        u_table[:,0] = T.flatten()
        u_table[:,1] = u.flatten()
        util.obs_to_csv(u_table, header=["t", "theta"], filename=f"data/PendulumParamSweepTest_{k}_{i}.csv")

    ## SPDE GMRF regression with unknown parameters
    # Perform iterative optimisation
    spde_rmses = np.empty(num_repeats)
    spde_mnlls = np.empty(num_repeats)
    for i, obs_dict in enumerate(datasets):
        model = nonlinear.NonlinearINLASPDERegressor(u, 1, dt, params_true, diff_op_gen, prior_mean_gen, logpdf_marginal_posterior,
                                                     mixing_coef=1, param_bounds=param_bounds, sampling_evec_scales=[3e-1, 1e-1],
                                                     sampling_threshold=0, params_true=params_true)
        try:
            spde_u0, spde_mean, spde_std = model.fit(obs_dict, obs_std, max_iter=max_iter, animated=False, calc_std=True, calc_mnll=False)
        except:
            params_opt[k_idx,i,:] = [float("inf"), float("inf")]
            rmses[k_idx,i] = float("inf")
            mnlls[k_idx,i] = float("inf")
            print("Optimisation failed")
            continue
        params_opt[k_idx,i,:] = model.params_opt
        rmses[k_idx,i] = metrics.rmse(spde_mean, u)
        mnlls[k_idx,i]  = -stats.norm.logpdf(u.flatten(), loc=spde_mean.flatten(), scale=spde_std.flatten()).mean()
        print(model.params_opt)
    print(f"RMSE = {np.mean(spde_rmses)} +- {np.std(spde_rmses)}")
    print(f"MNLL = {np.mean(spde_mnlls)} +- {np.std(spde_mnlls)}")

b_means = np.mean(params_opt[:,:,0], axis=1)
c_means = np.mean(params_opt[:,:,1], axis=1)
b_stds = np.std(params_opt[:,:,0], axis=1)
c_stds = np.std(params_opt[:,:,1], axis=1)

data_file = f"data/pendulum_estimation_sweep_opt_fine.pkl"
with open(data_file, 'wb') as f:
    pickle.dump({'windows': windows, 'popt': params_opt, 'b_means': b_means, 'b_stds': b_stds, 'c_means': c_means, 'c_stds': c_stds, 'ptrue': params_true, 'rmses': rmses, 'mnlls': mnlls}, f, protocol=pickle.HIGHEST_PROTOCOL)

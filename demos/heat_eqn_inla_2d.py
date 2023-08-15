import itertools
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.optimize import minimize, fmin_bfgs, fmin_l_bfgs_b
from scipy.special import loggamma
from sksparse.cholmod import cholesky
from findiff import FinDiff, PDE, BoundaryConditions
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 200

from spdeinf import inla, linear, metrics, plotting, util

# Define parameters of the stochastic heat equation
alpha_true = 0.15
W_amp = 0

# Define parameters of the parameter priors
t_obs_a = 5
t_obs_b = 0.1
tau_alpha = 1.
alpha_0 = 0
alpha_prior_mode = np.exp(alpha_0 - tau_alpha ** (-2))

## Generate data from the heat equation

# Create spatial discretisation
x_max = 2                       # Range of spatial domain
dx = 0.05                       # Spatial delta
N_x = int(x_max / dx) + 1       # Number of points in spatial discretisation
xx = np.linspace(0, x_max, N_x) # Spatial array

# Create temporal discretisation
t_max = 1                       # Range of temporal domain
dt = 0.05                       # Temporal delta
N_t = int(t_max / dt) + 1       # Number of points in temporal discretisation
tt = np.linspace(0, t_max, N_t) # Temporal array
shape = (N_x, N_t)

# Create test points
X, T = np.meshgrid(xx, tt, indexing='ij')
X_test = np.stack([X.flatten(), T.flatten()], axis=1)

# Define SPDE linear operator
def get_diff_op(alpha, dx, dt):
    diff_op_xx = FinDiff(0, dx, 2, acc=4)
    diff_op_t = FinDiff(1, dt, 1, acc=4)
    L = diff_op_t - alpha * diff_op_xx
    return L
diff_op_gen = lambda a: get_diff_op(a, dx, dt)
L = diff_op_gen(alpha_true)

# Define SPDE RHS
np.random.seed(13)
W = W_amp * np.random.randn(*shape)

# Set boundary conditions (Dirichlet)
bc = BoundaryConditions(shape)
for i in range(N_x):
    normal_pdf = np.exp(- ((xx[i] - 1.) / 0.2) ** 2)
    bc[i,0] = normal_pdf
bc[0,:] = 0
bc[-1,:] = 0

# Solve PDE
pde = PDE(L, W, bc)
u = pde.solve()
plt.imshow(u)
plt.xlabel('t')
plt.ylabel('x')
plt.show()

# Sample observations
obs_std = 1e-2
obs_count = 50
obs_dict = util.sample_observations(u, obs_count, obs_std)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_idxs_flat = shape[1] * obs_idxs[:,0] + obs_idxs[:,1]
obs_vals = np.array(list(obs_dict.values()))


# Define marginal parameter posterior
def _logpdf_marginal_posterior(a, t_obs, Q_u, Q_uy, Q_obs, mu_u, mu_uy, regularisation=1e-5):
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
    if a > 0:
        log_a = np.log(a)
        log_p_a = np.log(tau_alpha) - log_a - 0.5 * ((tau_alpha * (log_a - alpha_0)) ** 2 + log_2pi) # log-normal prior
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
    return logpdf

def logpdf_marginal_posterior(x, return_conditional_params=False):
    # Unpack parameters
    a = x[0]
    t_obs = x[1]

    # Fit linear model
    diff_op_guess = diff_op_gen(a)
    res = linear.fit_spde_gp(u, obs_dict, 1 / t_obs, diff_op_guess, prior_mean=0, calc_std=return_conditional_params,
                             calc_lml=False, include_boundary_cond=True, return_prior_precision=True,
                             return_posterior_precision=True, regularisation=1e-5)
    Q_u = res['prior_precision']
    Q_uy = res['posterior_precision']
    Q_obs = sparse.diags([t_obs ** 2], 0, shape=(obs_count, obs_count), format='csc')
    mu_u = np.zeros_like(u).flatten()
    mu_uy = res['posterior_mean'].flatten()
    # mu_uy = u.flatten()

    # Compute marginal posterior
    logpdf = _logpdf_marginal_posterior(a, t_obs, Q_u, Q_uy, Q_obs, mu_u, mu_uy)
    if return_conditional_params:
        return logpdf, res['posterior_mean'], res['posterior_var']
    return logpdf

# Generate INLA samples
samples, H_v = inla.sample_parameter_posterior(logpdf_marginal_posterior, [0.2, 200], sampling_evec_scales=[10, 0.015])
samples_x = samples[0]
samples_p = samples[1]
samples_mu = samples[2]
samples_var = samples[3]

# Get MAP parameters for convenience
alpha_map = samples_x[0,0]
t_obs_map = samples_x[0,1]
obs_std_map = 1 / t_obs_map

# Sweep marginal posterior for plotting
alpha_max = 1
alpha_count = 10
alpha_min = 1 / alpha_count
alphas = np.linspace(alpha_min , alpha_max, alpha_count)

t_obs_max = 500
t_obs_count = 20
t_obs_min = 1 / t_obs_count
t_obs_prior_mode = 40
t_obss = np.linspace(t_obs_min , t_obs_max, t_obs_count)

log_marg_post = np.empty((alpha_count, t_obs_count))
for i, a in tqdm(enumerate(alphas), total=alpha_count):
    for j, t_obs in enumerate(t_obss):
        log_marg_post[i,j] = logpdf_marginal_posterior([a, t_obs])

# Plot marg. parameter posterior
plt.contourf(t_obss, alphas, log_marg_post, levels=50)
plt.scatter(t_obs_map, alpha_map, c='r', marker='x', label="MAP $\\theta$")
plt.scatter(t_obs_prior_mode, alpha_prior_mode, c='b', marker='x', label="Prior mode $\\theta$")
plt.scatter(1 / obs_std, alpha_true, c='m', marker='x', label="True $\\theta$")
plt.scatter(samples_x[:,1], samples_x[:,0], s=5, c='k', label="Sampled points")
# plt.quiver(*H_v_origins, H_v[1,:], H_v[0,:], width=0.005, scale=8, label="Eigenvectors of Hessian")
plt.xlabel('$\\tau_{obs}$')
plt.ylabel('$\\alpha$')
plt.title('$\\log \\widetilde{p}(\\theta | y)$')
plt.legend()
plt.tight_layout()
plt.savefig("figures/heat_eqn/heat_eqn_inla_parameter_posterior.png", dpi=200)
plt.show()

# Compute posterior marginal for field
posterior_mean_marg, posterior_std_marg = inla.compute_field_posterior_stats(samples)
print(f'Marginal mean MSE={metrics.mse(u, posterior_mean_marg)}')

# Fit with MAP estimate of alpha and obs noise
diff_op_map = diff_op_gen(alpha_map)
res = linear.fit_spde_gp(u, obs_dict, obs_std_map, diff_op_map, calc_std=True, calc_lml=False, include_boundary_cond=True)
posterior_mean_map = res['posterior_mean']
posterior_std_map = res['posterior_std']
print(f'MAP alpha={alpha_map}, MSE={metrics.mse(u, posterior_mean_map)}')

# Fit with true alpha and obs noise
diff_op_true = diff_op_gen(alpha_true)
res = linear.fit_spde_gp(u, obs_dict, obs_std, diff_op_true, calc_std=True, calc_lml=False, include_boundary_cond=True)
posterior_mean_true = res['posterior_mean']
posterior_std_true = res['posterior_std']
print(f'True alpha={alpha_true}, MSE={metrics.mse(u, posterior_mean_true)}')

# Plot results
plot_kwargs = {
        'mean_vmin': min(posterior_mean_map.min(), posterior_mean_true.min()),
        'mean_vmax': max(posterior_mean_map.max(), posterior_mean_true.max()),
        'std_vmin': 0,
        'std_vmax': max(posterior_std_map.max(), posterior_std_true.max()),
        'diff_vmin': -0.3,
        'diff_vmax': 0.3,
        }
plotting.plot_gp_2d(u, posterior_mean_map, posterior_std_map, obs_idxs, 'figures/heat_eqn/heat_eqn_inla_alpha_map.png', **plot_kwargs)
plotting.plot_gp_2d(u, posterior_mean_true, posterior_std_true, obs_idxs, 'figures/heat_eqn/heat_eqn_inla_alpha_true.png', **plot_kwargs)
plotting.plot_gp_2d(u, posterior_mean_marg, posterior_std_marg, obs_idxs, 'figures/heat_eqn/heat_eqn_inla_alpha_marg.png', **plot_kwargs)

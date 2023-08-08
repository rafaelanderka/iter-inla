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

from spdeinf import linear, metrics, util, plotting

## Generate data from the heat equation

# Define parameters of the stochastic heat equation
alpha_true = 0.2
W_amp = 0

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
obs_noise = 1e-2
obs_count = 50
obs_dict = util.sample_observations(u, obs_count, obs_noise)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_idxs_flat = shape[1] * obs_idxs[:,0] + obs_idxs[:,1]
obs_vals = np.array(list(obs_dict.values()))

# Test if flat indices are correct
test_fill = np.zeros_like(u).flatten()
test_fill[obs_idxs_flat] = 1
print(test_fill.reshape(shape))

# Define marginal posterior for alpha
def _logpdf_marginal_posterior(a, t_obs, t_a, a_0, Q_u, Q_uy, Q_obs, mu_u, mu_uy, obs_vals, obs_idxs, t_obs_a=5, t_obs_b=0.1, regularisation=1e-5):
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
        log_p_a = np.log(t_a) - log_a - 0.5 * ((t_a * (log_a - a_0)) ** 2 + log_2pi) # log-normal prior
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
    diff_obs_mu_uy = obs_vals - mu_uy[obs_idxs]
    log_p_yua = 0.5 * (Q_obs_logdet - diff_obs_mu_uy.T @ Q_obs @ diff_obs_mu_uy - N * log_2pi)

    # Compute full conditional terms
    log_p_uya = -0.5 * (Q_uy_logdet - M * log_2pi)

    return np.array([log_p_a, log_p_t_obs, log_p_ua, log_p_yua, log_p_uya])

def logpdf_marginal_posterior(a, t_obs, diff_op_gen, u, obs_dict, obs_noise, obs_count, obs_vals, obs_idxs_flat, tau_alpha, alpha_0):
    diff_op_guess = diff_op_gen(a)
    res = linear.fit_spde_gp(u, obs_dict, 1 / t_obs, diff_op_guess, calc_std=False, calc_lml=False, include_boundary_cond=True,
                             return_prior_precision=True, return_posterior_precision=True, regularisation=1e-5)
    Q_u = res['prior_precision']
    Q_uy = res['posterior_precision']
    Q_obs = sparse.diags([t_obs ** 2], 0, shape=(obs_count, obs_count), format='csc')
    mu_u = np.zeros_like(u).flatten()
    mu_uy = res['posterior_mean'].flatten()
    return _logpdf_marginal_posterior(a, t_obs, tau_alpha, alpha_0, Q_u, Q_uy, Q_obs, mu_u, mu_uy, obs_vals, obs_idxs_flat)


# Sweep marginal posterior
tau_alpha = 1.
alpha_0 = 0
alpha_max = 1
alpha_count = 10
alpha_min = 1 / alpha_count
alpha_prior_mode = np.exp(alpha_0 - tau_alpha ** (-2))
alphas = np.linspace(alpha_min , alpha_max, alpha_count)

t_obs_max = 1000
t_obs_count = 20
t_obs_min = 1 / t_obs_count
t_obs_prior_mode = 40
t_obss = np.linspace(t_obs_min , t_obs_max, t_obs_count)

log_marg_post = np.empty((alpha_count, t_obs_count, 5))
for i, a in tqdm(enumerate(alphas), total=alpha_count):
    for j, t_obs in enumerate(t_obss):
        log_marg_post[i,j,:] = logpdf_marginal_posterior(a, t_obs, diff_op_gen, u, obs_dict, obs_noise, obs_count, obs_vals, obs_idxs_flat, tau_alpha, alpha_0)

# Find MAP from parameter marginal posterior
neg_logpdf_mp = lambda x: -np.sum(logpdf_marginal_posterior(x[0], x[1], diff_op_gen, u, obs_dict, obs_noise, obs_count, obs_vals, obs_idxs_flat, tau_alpha, alpha_0), axis=-1)
opt = minimize(fun=neg_logpdf_mp, x0=[0.2, 200], method="Nelder-Mead")
print(opt)
alpha_map = opt["x"][0]
t_obs_map = opt["x"][1]
obs_noise_map = 1 / t_obs_map

# Calculate hessian
def hessian(fun, x0, epsilon=1e-3):
    # epsilon = np.sqrt(np.finfo(float).eps)
    n = len(x0)
    hess = np.zeros((n, n))
    for i in range(n):
        x1 = np.array(x0, copy=True)
        x2 = np.array(x0, copy=True)
        x1[i] += epsilon
        x2[i] -= epsilon
        hess[i, i] = (fun(x1) - 2 * fun(x0) + fun(x2)) / (epsilon ** 2)
        for j in range(i+1, n):
            x1[i] -= epsilon
            x1[j] += epsilon
            x2[i] += epsilon
            x2[j] -= epsilon
            hess[i, j] = (fun(x1) - fun(x2)) / (4 * epsilon ** 2)
            hess[j, i] = hess[i, j]
    return hess

# Calculate eigenvectors of Hessian along which to sample
H = hessian(neg_logpdf_mp, [alpha_map, t_obs_map]) # H is (M, M)
H_w, H_v = eigh(H) # Note H_v is (M, N)
H_v = -H_v # Flip just for intuition
H_v_origins = np.array([[t_obs_map], [alpha_map]]).repeat(2, axis=1)
print(H_w)
print(H_v)

# Sample along eigenvectors
x0 = np.array([alpha_map, t_obs_map])
p0 = -neg_logpdf_mp(x0)
thresh = 5
step_size = 2
samples_x = []
samples_p = []
evec_scales = np.array([10, 0.015])
N_evec = H_v.shape[1]
ranges = []
for i, evec in enumerate(H_v.T):
    r = []
    for dir in [-1, 1]:
        offset = 0
        xS = x0.copy()
        while p0 + neg_logpdf_mp(xS) < thresh:
            samples_x.append(xS.copy())
            samples_p.append(-neg_logpdf_mp(xS))
            offset += dir
            xS = x0 + offset * step_size * evec_scales[i] * evec
            # print(step_size * dir * evec)
        r.append(offset - dir)
    ranges.append(r)
print(ranges)

def generate_combinations(ranges):
    # Convert the 2-tuple ranges to aranges
    range_generators = [range(start, end+1) for start, end in ranges]
    # Compute the cartesian product of the lists
    return itertools.product(*range_generators)

for offset in generate_combinations(ranges):
    xS = x0 + step_size * (evec_scales * offset * H_v).sum(axis=1)
    pS = -neg_logpdf_mp(xS)
    if p0 - pS < thresh:
        samples_x.append(xS)
        samples_p.append(pS)

samples_x = np.array(samples_x)
samples_p = np.array(samples_p)

# Plot marg. posterior
plt.contourf(t_obss, alphas, log_marg_post.sum(axis=-1), levels=50)
plt.scatter(t_obs_map, alpha_map, c='r', marker='x', label="MAP $\\theta$")
plt.scatter(t_obs_prior_mode, alpha_prior_mode, c='b', marker='x', label="Prior mode $\\theta$")
plt.scatter(1 / obs_noise, alpha_true, c='m', marker='x', label="True $\\theta$")
plt.scatter(samples_x[:,1], samples_x[:,0], label="Sampled points")
plt.quiver(*H_v_origins, H_v[1,:], H_v[0,:], width=0.005, scale=8, label="Eigenvectors of Hessian")
plt.xlabel('$\\tau_{obs}$')
plt.ylabel('$\\alpha$')
plt.title('$\\log \\widetilde{p}(\\theta | y)$')
plt.legend()
plt.show()

# Fit with MAP estimate of alpha and obs noise
diff_op_map = diff_op_gen(alpha_map)
res = linear.fit_spde_gp(u, obs_dict, obs_noise_map, diff_op_map, calc_std=True, calc_lml=False, include_boundary_cond=True)
posterior_mean_map = res['posterior_mean']
posterior_std_map = res['posterior_std']
print(f'MAP alpha={alpha_map}, MSE={metrics.mse(u, posterior_mean_map)}')

# Fit with true alpha and obs noise
diff_op_true = diff_op_gen(alpha_true)
res = linear.fit_spde_gp(u, obs_dict, obs_noise, diff_op_true, calc_std=True, calc_lml=False, include_boundary_cond=True)
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

import numpy as np
from findiff import FinDiff, PDE, BoundaryConditions

from spdeinf import linear, plotting, metrics, util

# Generate data from heat equation

# Set config params
grid_size = 80
time_size = 40
W_amp = 0

# Create domain
t = np.linspace(0, 1, time_size)
x = np.linspace(0, 1, grid_size)
X, T = np.meshgrid(t, x, indexing='ij')
X_test = np.stack([T.flatten(), X.flatten()], axis=1)
shape = (time_size, grid_size)
dt = t[1]-t[0]
dx = x[1]-x[0]

# Define SPDE LHS
alpha = 0.05
diff_op = FinDiff(0, dt, 1) - alpha * FinDiff(1, dx, 2)

# Define SPDE RHS
np.random.seed(13)
W = W_amp * np.random.randn(*shape)

# Set boundary conditions (Dirichlet)
bc = BoundaryConditions(shape)
for i in range(grid_size):
    normal_pdf = np.exp(- ((x[i] - 0.5) / 0.1) ** 2)
    bc[0,i] = normal_pdf
bc[:,0] = 0
bc[:,-1] = 0

# Solve PDE
pde = PDE(diff_op, W, bc)
u = pde.solve()

# Sample observations
obs_noise = 1e-4
obs_count = 100
obs_dict = util.sample_observations(u, obs_count, obs_noise)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)

# Fit with PDE prior
for a in np.linspace(0.01, 0.1, 10):
    diff_op_guess = FinDiff(0, dt, 1) - a * FinDiff(1, dx, 2)
    res = linear.fit_spde_gp(u, obs_dict, obs_noise, diff_op_guess, calc_std=True, calc_lml=True)
    posterior_mean_pde = res['posterior_mean']
    posterior_std_pde = res['posterior_std']
    log_marginal_likelihood = res['log_marginal_likelihood']
    print(f'a={a:.2f}, MSE={metrics.mse(u, posterior_mean_pde)}, LML={log_marginal_likelihood}')
plotting.plot_gp_2d(u.T, posterior_mean_pde.T, posterior_std_pde.T, util.swap_cols(obs_idxs), 'figures/heat_eqn_test_pde.png',
                  mean_vmin=-0.05, mean_vmax=1, std_vmin=0, std_vmax=0.06,
                  diff_vmin=-0.3, diff_vmax=0.2)

# # Fit with RBF prior
# posterior_mean_rbf, posterior_std_rbf = linear.fit_rbf_gp(u, obs_dict, X_test, dx, dt, obs_noise)
# print(metrics.mse(u, posterior_mean_rbf))
# plotting.plot_gp_2d(u.T, posterior_mean_rbf.T, posterior_std_rbf.T, util.swap_cols(obs_idxs), 'figures/heat_eqn_test_rbf.png',
#                   mean_vmin=-0.05, mean_vmax=1, std_vmin=0, std_vmax=0.06, diff_vmin=-0.2, diff_vmax=0.055)

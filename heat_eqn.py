import numpy as np
from findiff import FinDiff, PDE, BoundaryConditions

import linear

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
obs_dict = linear.sample_observations(u, obs_count, obs_noise)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)

# Fit with PDE prior
for a in np.linspace(0.01, 0.1, 10):
    diff_op_guess = FinDiff(0, dt, 1) - a * FinDiff(1, dx, 2)
    posterior_mean_pde, posterior_std_pde, log_marginal_likelihood = linear.fit_spde_grf(u, obs_dict, X_test, dx, dt, obs_noise, diff_op_guess)
    print(f'a={a:.2f}, MSE={linear.mse(u, posterior_mean_pde)}, LML={log_marginal_likelihood}')
# linear.plot_gp_2d(u.T, posterior_mean_pde.T, posterior_std_pde.T, linear.swap_cols(obs_idxs), 'figures/heat_eqn_test_pde.png',
#                   mean_vmin=-0.05, mean_vmax=1, std_vmin=0, std_vmax=0.06,
#                   diff_vmin=-0.3, diff_vmax=0.2)

# # Fit with RBF prior
# posterior_mean_rbf, posterior_std_rbf = linear.fit_rbf_gp(u, obs_dict, X_test, dx, dt, obs_noise)
# print(linear.mse(u, posterior_mean_rbf))
# linear.plot_gp_2d(u.T, posterior_mean_rbf.T, posterior_std_rbf.T, linear.swap_cols(obs_idxs), 'figures/heat_eqn_test_rbf.png',
#                   mean_vmin=-0.05, mean_vmax=1, std_vmin=0, std_vmax=0.06, diff_vmin=-0.2, diff_vmax=0.055)

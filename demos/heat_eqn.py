import os
import sys
import numpy as np
import jax
from findiff import FinDiff, PDE, BoundaryConditions

from spdeinf import linear, plotting, metrics, util

## Generate data from heat equation

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
diff_op_t = FinDiff(0, dt, 1)
diff_op_x = FinDiff(1, dx, 2)
L = diff_op_t - alpha * diff_op_x

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
pde = PDE(L, W, bc)
u = pde.solve()

# Sample observations
obs_noise = 1e-4
obs_count = 100
obs_dict = util.sample_observations(u, obs_count, obs_noise)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)

# Construct fin. diff. matrices
diff_mat_t = util.scipy2jax_csr(diff_op_t.matrix(shape))
diff_mat_x = util.scipy2jax_csr(diff_op_x.matrix(shape))

def lml(a):
    prior_precision = diff_mat_t - a * diff_mat_x
    mean, std, log_ml = linear._fit_grf(u, obs_dict, obs_noise, prior_precision)
    return log_ml, (mean, std)

lml_with_grad = jax.value_and_grad(lml)

# Fit with PDE prior
lr = 0.01
alpha = 0.1
for i in range(10):
    (log_ml, (mean, std)), alpha_grad = lml_with_grad(alpha)
    alpha -= lr * alpha_grad
    print(f'a={alpha:.2f}, MSE={metrics.mse(u, mean):.8f}, LML={log_ml:.2f}')
# plotting.plot_gp_2d(u.T, posterior_mean_pde.T, posterior_std_pde.T, util.swap_cols(obs_idxs), 'figures/heat_eqn_test_pde.png',
#                   mean_vmin=-0.05, mean_vmax=1, std_vmin=0, std_vmax=0.06,
#                   diff_vmin=-0.3, diff_vmax=0.2)

# # Fit with RBF prior
# posterior_mean_rbf, posterior_std_rbf = linear.fit_rbf_gp(u, obs_dict, X_test, dx, dt, obs_noise)
# print(metrics.mse(u, posterior_mean_rbf))
# plotting.plot_gp_2d(u.T, posterior_mean_rbf.T, posterior_std_rbf.T, util.swap_cols(obs_idxs), 'figures/heat_eqn_test_rbf.png',
#                   mean_vmin=-0.05, mean_vmax=1, std_vmin=0, std_vmax=0.06, diff_vmin=-0.2, diff_vmax=0.055)

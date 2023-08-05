import numpy as np
from findiff import FinDiff, PDE, BoundaryConditions
import matplotlib.pyplot as plt

from spdeinf import linear, metrics, util, plotting

## Generate data from the heat equation

# Define parameters of the stochastic heat equation
alpha = 0.05
W_amp = 0

# Create spatial discretisation
x_max = 1                       # Range of spatial domain
dx = 0.1                        # Spatial delta
N_x = int(x_max / dx) + 1       # Number of points in spatial discretisation
xx = np.linspace(0, x_max, N_x) # Spatial array

# Create temporal discretisation
t_max = 1                       # Range of temporal domain
dt = 0.1                        # Temporal delta
N_t = int(t_max / dt) + 1       # Number of points in temporal discretisation
tt = np.linspace(0, t_max, N_t) # Temporal array
shape = (N_x, N_t)

# Create test points
X, T = np.meshgrid(xx, tt, indexing='ij')
X_test = np.stack([X.flatten(), T.flatten()], axis=1)

# Define SPDE linear operator
diff_op_xx = FinDiff(0, dx, 2)
diff_op_t = FinDiff(1, dt, 1)
L = diff_op_t - alpha * diff_op_xx

# Define SPDE RHS
np.random.seed(13)
W = W_amp * np.random.randn(*shape)

# Set boundary conditions (Dirichlet)
bc = BoundaryConditions(shape)
for i in range(N_x):
    normal_pdf = np.exp(- ((xx[i] - 0.5) / 0.1) ** 2)
    bc[i,0] = normal_pdf
bc[0,:] = 0
bc[-1,:] = 0

# Solve PDE
pde = PDE(L, W, bc)
u = pde.solve()
plt.imshow(u)
plt.show()

# Sample observations
obs_noise = 1e-2
obs_count = 10
obs_dict = util.sample_observations(u, obs_count, obs_noise)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)

# Fit with SPDE prior
diff_op_guess = diff_op_t - alpha * diff_op_xx
res = linear.fit_spde_gp(u, obs_dict, obs_noise, diff_op_guess, calc_std=True, calc_lml=False)
posterior_mean_pde = res['posterior_mean']
posterior_std_pde = res['posterior_std']
print(f'SPDE prior MSE={metrics.mse(u, posterior_mean_pde)}')

# Fit with RBF prior
posterior_mean_rbf, posterior_std_rbf = linear.fit_rbf_gp(u, obs_dict, X_test, dx, dt, obs_noise)
print(f'RBF prior MSE={metrics.mse(u, posterior_mean_rbf)}')

# Plot results
plot_kwargs = {
        'mean_vmin': u.min(),
        'mean_vmax': u.max(),
        'std_vmin': 0,
        'std_vmax': max(posterior_std_pde.max(), posterior_std_rbf.max()),
        'diff_vmin': -0.3,
        'diff_vmax': 0.8,
        }
plotting.plot_gp_2d(u, posterior_mean_pde, posterior_std_pde, obs_idxs, 'figures/heat_eqn/heat_eqn_test_pde.png', **plot_kwargs)
plotting.plot_gp_2d(u, posterior_mean_rbf, posterior_std_rbf, obs_idxs, 'figures/heat_eqn/heat_eqn_test_rbf.png', **plot_kwargs)

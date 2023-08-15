import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from findiff import FinDiff, Coef, Identity
from sksparse.cholmod import cholesky

from spdeinf import nonlinear, linear, plotting, util, metrics

## Generate data from Burger's equation

# Define parameters of Burgers' equation
mu = 1 
nu = 0.03 # Kinematic viscosity coefficient

# Create spatial discretisation
L_x = 10                      # Range of spatial domain
dx = 0.1                      # Spatial delta
N_x = int(L_x / dx) + 1       # Number of points in spatial discretisation
X = np.linspace(0, L_x, N_x)  # Spatial array

# Create temporal discretisation
L_t = 8                       # Range of temporal domain
dt = 0.1                      # Temporal delta
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
u = odeint(burgers_odes, u0, T, args=(k, mu, nu,), mxstep=5000).T


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

def get_prior_mean(u0, diff_op_gen):
    """
    Calculates current prior mean.
    """
    partial_x = FinDiff(0, dx, 1)
    u0_x = partial_x(u0)
    diff_op = diff_op_gen(u0)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (u0 * u0_x).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u: get_diff_op(u, dx, dt, nu)
prior_mean_gen = lambda u: get_prior_mean(u, diff_op_gen)


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

diff_op_gen_naive = lambda u: get_diff_op_naive(u, dx, dt, nu)
prior_mean_gen_naive = lambda u: get_prior_mean_naive(u, diff_op_gen)

## Fit GP with non-linear SPDE prior from Burgers' equation

# Sample observations
obs_std = 1e-2
obs_count = 100
obs_dict = util.sample_observations(u, obs_count, obs_std, extent=(None, None, 0, 20))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
print("Number of observations:", obs_idxs.shape[0])

# Perform iterative optimisation
max_iter = 50
model = nonlinear.NonlinearSPDERegressor(u, dx, dt, diff_op_gen, prior_mean_gen, mixing_coef=0.5)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True)
iter_count = len(model.mse_hist)

# Check prior covariance
diff_op_init = diff_op_gen(np.zeros_like(u))
diff_op_final = diff_op_gen(model.posterior_mean)

L_init = util.operator_to_matrix(diff_op_init, u.shape, interior_only=False)
L_final = util.operator_to_matrix(diff_op_final, u.shape, interior_only=False)
LL_init = L_init.T @ L_init
LL_final = L_final.T @ L_final
LL_init_chol = cholesky(LL_init + identity(LL_init.shape[0]))
LL_final_chol = cholesky(LL_final + identity(LL_final.shape[0]))
prior_init_std = np.sqrt(LL_init_chol.spinv().diagonal().reshape(u.shape))
prior_final_std = np.sqrt(LL_final_chol.spinv().diagonal().reshape(u.shape))

fig, ax = plt.subplots(1, 2)
im_init = ax[0].imshow(prior_init_std, origin="lower")
im_final = ax[1].imshow(prior_final_std, origin="lower")
ax[0].set_xlabel('x')
ax[0].set_ylabel('t')
fig.colorbar(im_init)
fig.colorbar(im_final)
plt.show()

# Fit with naive linearisation
model_naive = nonlinear.NonlinearSPDERegressor(u, dx, dt, diff_op_gen_naive, prior_mean_gen_naive)
model_naive.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True)
iter_count_naive = len(model_naive.mse_hist)

# Plot convergence history
plt.plot(np.arange(1, iter_count + 1), model.mse_hist, label="Linearisation via expansion")
plt.plot(np.arange(1, iter_count_naive + 1), model_naive.mse_hist, label="Naive linearisation")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.xticks(np.arange(2, max(iter_count, iter_count_naive) + 1, 2))
plt.legend()
plt.savefig("figures/burgers_eqn/mse_conv.png", dpi=200)
plt.show()

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

# Save animation
print("Saving animation...")
model.save_animation("figures/burgers_eqn/burgers_eqn_iter_animation.gif", fps=10)

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from findiff import FinDiff, Coef, Identity
from sksparse.cholmod import cholesky

from spdeinf import nonlinear, linear, plotting, util, metrics

## Generate data from Burger's equation
np.random.seed(0)

# Define parameters of Burgers' equation
nu = 0.01 # Kinematic viscosity coefficient

# Create spatial discretisation
L_x = 1                      # Range of spatial domain
dx = 0.01                     # Spatial delta
N_x = int(2 * L_x / dx) + 1       # Number of points in spatial discretisation
xx = np.linspace(-L_x, L_x, N_x)  # Spatial array

# Create temporal discretisation
L_t = 1                       # Range of temporal domain
dt = 0.01                     # Temporal delta
N_t = int(L_t / dt) + 1       # Number of points in temporal discretisation
tt = np.linspace(0, L_t, N_t)  # Temporal array

# Define wave number discretization
k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

# Define the initial condition    
# u0 = np.exp(-(xx - 0.3)**2 / 0.1)
u0 = -np.sin(np.pi * (xx))

def burgers_odes(u, t, k, nu):
    """
    Construct system of ODEs for Burgers' equation using pseudo-spectral method of lines
    """
    # Take x derivatives in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j * k * u_hat
    u_hat_xx = -k**2 * u_hat

    # Transform back to spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)

    # Assemble ODE
    u_t = -u * u_x + nu * u_xx
    return u_t.real

# Solve system of ODEs
u = odeint(burgers_odes, u0, tt, args=(k, nu,), mxstep=5000).T

# Sample observations
obs_std = 1e-9
obs_count_1 = 128
obs_count_2 = 0
obs_loc_1 = np.where(tt == 0.1)[0][0]
obs_loc_2 = np.where(tt == 0.75)[0][0]
obs_dict = util.sample_observations(u, obs_count_1, obs_std, extent=(None, None, 0, obs_loc_1))
obs_dict.update(util.sample_observations(u, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()))
print("Number of observations:", obs_idxs.shape[0])

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dx, dt, nu):
    """
    Constructs current linearised differential operator.
    """
    partial_t = FinDiff(1, dt, 1, acc=2)
    partial_x = FinDiff(0, dx, 1, acc=2, periodic=True)
    partial_xx = FinDiff(0, dx, 2, acc=2, periodic=True)
    u0_x = partial_x(u0)
    diff_op = partial_t + Coef(u0) * partial_x - Coef(nu) * partial_xx + Coef(u0_x) * Identity()
    return diff_op

def get_prior_mean(u0, diff_op_gen):
    """
    Calculates current prior mean.
    """
    partial_x = FinDiff(0, dx, 1, acc=2, periodic=True)
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
    partial_t = FinDiff(1, dt, 1, acc=2)
    partial_x = FinDiff(0, dx, 1, acc=2, periodic=True)
    partial_xx = FinDiff(0, dx, 2, acc=2, periodic=True)
    diff_op = partial_t + Coef(u0) * partial_x - Coef(nu) * partial_xx
    return diff_op

def get_prior_mean_naive(u0, diff_op_gen):
    return np.zeros_like(u0)

diff_op_gen_naive = lambda u: get_diff_op_naive(u, dx, dt, nu)
prior_mean_gen_naive = lambda u: get_prior_mean_naive(u, diff_op_gen)

## Fit GP with non-linear SPDE prior from Burgers' equation

# Perform iterative optimisation
max_iter = 20
model = nonlinear.NonlinearSPDERegressor(u, dx, dt, diff_op_gen, prior_mean_gen, mixing_coef=0.5)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)
iter_count = len(model.mse_hist)

# Fit with naive linearisation
model_naive = nonlinear.NonlinearSPDERegressor(u, dx, dt, diff_op_gen_naive, prior_mean_gen_naive, mixing_coef=0.2)
model_naive.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)
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

# Save animation
print("Saving animation...")
model.save_animation("figures/burgers_eqn/burgers_eqn_iter_animation.gif", fps=10)

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from findiff import FinDiff, Coef, Identity

from spdeinf import nonlinear, util

## Generate data from nonlinear damped pendulum eqn.

# Define parameters of damped pendulum
b = 0.5 
c = 5.

# Create temporal discretisation
L_t = 25                      # Duration of simulation [s]
dt = 0.1                      # Infinitesimal time
N_t = int(L_t / dt) + 1       # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array

# Define the initial condition    
u0 = [np.pi - 0.1, 0.]

# Define corresponding system of ODEs
def pend(u, t, b, c):
    theta, omega = u
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

# Solve system of ODEs
u = odeint(pend, u0, T, args=(b, c,))

# For our purposes we only need the solution for the pendulum angle, theta
u = u[:, 0].reshape(1, -1)

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dt, b, c):
    """
    Constructs current linearised differential operator.
    """
    partial_t = FinDiff(1, dt, 1)
    partial_tt = FinDiff(1, dt, 2)
    u0_cos = np.cos(u0)
    diff_op = partial_tt + Coef(b) * partial_t + Coef(c * u0_cos) * Identity()
    return diff_op

def get_prior_mean(u0, diff_op_gen, c):
    """
    Calculates current prior mean.
    """
    u0_cos = np.cos(u0)
    u0_sin = np.sin(u0)
    diff_op = diff_op_gen(u0)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (c * (u0 * u0_cos - u0_sin)).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u: get_diff_op(u, dt, b, c)
prior_mean_gen = lambda u: get_prior_mean(u, diff_op_gen, c)

## Fit GP with non-linear SPDE prior from damped pendulum

# Sample observations
obs_noise = 1e-1
obs_count = 50
obs_lim = 50
obs_dict = util.sample_observations(u, obs_count, obs_noise, extent=(None, None, 0, obs_lim))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()))
print("Number of observations:", obs_idxs.shape[0])

# Perform iterative optimisation
max_iter = 40
model = nonlinear.NonlinearSPDERegressor(u, 1, dt, diff_op_gen, prior_mean_gen, mixing_coef=0.5)
model.fit(obs_dict, obs_noise, max_iter=max_iter, animated=False, calc_std=True)
iter_count = len(model.mse_hist)

# Plot fit
plt.plot(T, u.squeeze(), "b", label="Ground truth")
plt.plot(T, model.posterior_mean.squeeze(), "k", label="Posterior mean")
plt.plot(T, model.posterior_mean.squeeze() + model.posterior_std.squeeze(), "--", color="grey", label="Posterior std.")
plt.plot(T, model.posterior_mean.squeeze() - model.posterior_std.squeeze(), "--", color="grey")
plt.axvline(dt * obs_lim, color='grey', ls=':')
plt.scatter(dt * obs_idxs[:,1], obs_vals, c="r", marker="x", label="Observations")
plt.legend(loc="best")
plt.xlabel("t")
plt.ylabel("$\\theta$")
plt.savefig("figures/pendulum/pendulum_fit.png", dpi=200)
plt.show()

# Plot convergence history
plt.plot(np.arange(1, iter_count + 1), model.mse_hist, label="Linearisation via expansion")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.xticks(np.arange(2, iter_count + 1, 2))
plt.legend()
plt.savefig("figures/pendulum/mse_conv.png", dpi=200)
plt.show()

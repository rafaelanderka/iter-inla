import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from findiff import FinDiff, Coef, Identity

from spdeinf import util
from spdeinf.nonlinear import SPDEDynamics, IterativeRegressor

# Set seed
np.random.seed(0)

####################################
# Generate data from Burgers' eqn. #
####################################

# Define parameters of Burgers' equation and model
nu = 0.01 # Kinematic viscosity coefficient
obs_std = 1e-9 # Observation noise
params = np.array([nu, obs_std])

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

# Define wave number discretisation
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
obs_count_1 = 128
obs_count_2 = 0
obs_loc_1 = np.where(tt == 0.1)[0][0]
obs_loc_2 = np.where(tt == 0.75)[0][0]
obs_dict = util.sample_observations(u, obs_count_1, obs_std, extent=(None, None, 0, obs_loc_1))
obs_dict.update(util.sample_observations(u, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()))
print("Number of observations:", obs_idxs.shape[0])

#################################
# Define Burgers' eqn. dynamics #
#################################

class BurgersDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. nu
    1. observation noise
    """

    def __init__(self, dx, dt) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt

    def get_diff_op(self, u0, params, **kwargs):
        """
        Construct current linearised differential operator.
        """
        nu, _ = params
        partial_t = FinDiff(1, dt, 1, acc=2)
        partial_x = FinDiff(0, dx, 1, acc=2, periodic=True)
        partial_xx = FinDiff(0, dx, 2, acc=2, periodic=True)
        u0_x = partial_x(u0)
        diff_op = partial_t + Coef(u0) * partial_x - Coef(nu) * partial_xx + Coef(u0_x) * Identity()
        return diff_op

    def get_prior_precision(self, u0, params, **kwargs):
        """
        Calculate current prior precision.
        """
        diff_op_guess = self.get_diff_op(u0, params, **kwargs)
        L = util.operator_to_matrix(diff_op_guess, u0.shape, interior_only=False)
        # prior_precision = (self.dt * self.dx / params[2]**2) * (L.T @ L)
        # prior_precision = self.dt * self.dx * (L.T @ L)
        prior_precision = L.T @ L
        return prior_precision

    def get_prior_mean(self, u0, params, **kwargs):
        """
        Calculate current prior mean.
        """
        # Construct linearisation remainder term
        partial_x = FinDiff(0, dx, 1, acc=2, periodic=True)
        u0_x = partial_x(u0)
        remainder = u0 * u0_x

        # Construct diff. op.
        diff_op = self.get_diff_op(u0, params, **kwargs)
        L = diff_op.matrix(u0.shape)

        # Compute prior mean
        prior_mean = spsolve(L.T @ L, L.T @ remainder.flatten())
        return prior_mean.reshape(u0.shape)

    def get_obs_noise(self, params, **kwargs):
        """
        Get observation noise (standard deviation).
        """
        return params[1]

dynamics = BurgersDynamics(dx, dt)

##########################################
# Fit model with iterative linearistaion #
##########################################

max_iter = 20
model = IterativeRegressor(u, dynamics, mixing_coef=0.1)
model.fit(obs_dict, params, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)

############
# Plot fit #
############

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_mean, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="RdBu_r")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/burgers_eqn/burgers_spde.pdf", transparent=True)

plt.figure(figsize=(3,3))
plt.imshow(model.posterior_std, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/burgers_eqn/burgers_spde_std.pdf", transparent=True)
plt.show()

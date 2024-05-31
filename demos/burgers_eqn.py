import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from findiff import FinDiff, Coef, Identity

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from iinla import util
from iinla.nonlinear import SPDEDynamics, IterativeRegressor

# Set seed
np.random.seed(0)

####################################
# Generate data from Burgers' eqn. #
####################################

# Define parameters of Burgers' equation and model
nu = 0.005 # Kinematic viscosity coefficient
obs_std = 1e-2 # Observation noise
params = np.array([nu, obs_std])

# Create spatial discretisation
L_x = 1                           # Range of spatial domain
dx = 0.01                        # Spatial delta
N_x = int(2 * L_x / dx)          # Number of points in spatial discretisation
xx = np.linspace(-L_x, L_x - dx, N_x)  # Spatial array

# Create temporal discretisation
L_t = 2                      # Range of temporal domain
dt = 0.01                      # Temporal delta
N_t = int(L_t / dt)            # Number of points in temporal discretisation
tt = np.linspace(0, L_t - dt, N_t)  # Temporal array

# Define wave number discretisation
k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

# Define the initial condition    
u0 = np.exp(-(xx - 0.3)**2 / 0.1)
# u0 = -np.sin(np.pi * (xx))

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
obs_count_1 = 64
obs_count_2 = 20
obs_loc_1 = np.where(tt == 0)[0][0]
obs_loc_2 = np.where(tt == 0.2)[0][0]
obs_dict = util.sample_observations(u, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_2))
# obs_dict.update(util.sample_observations(u, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()))
obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
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

    def _update_diff_op(self):
        """
        Construct current linearised differential operator.
        """
        nu = self._params[0]
        partial_t = FinDiff(1, dt, 1, acc=2)
        partial_x = FinDiff(0, dx, 1, acc=2, periodic=True)
        partial_xx = FinDiff(0, dx, 2, acc=2, periodic=True)
        u0_x = partial_x(self._u0)
        diff_op = partial_t + Coef(self._u0) * partial_x - Coef(nu) * partial_xx + Coef(u0_x) * Identity()
        return diff_op

    def _update_prior_precision(self):
        """
        Calculate current prior precision.
        """
        # prior_precision = (self.dt * self.dx / self._params[2]**2) * (self._L.T @ self._L)
        # prior_precision = self.dt * self.dx * (self._L.T @ self._L)
        prior_precision = self._L.T @ self._L
        return prior_precision

    def _update_prior_mean(self):
        """
        Calculate current prior mean.
        """
        # Construct linearisation remainder term
        partial_x = FinDiff(0, dx, 1, acc=2, periodic=True)
        u0_x = partial_x(self._u0)
        remainder = self._u0 * u0_x

        # Compute prior mean
        prior_mean = spsolve(self._L.T @ self._L, self._L.T @ remainder.flatten())
        return prior_mean.reshape(self._u0.shape)

    def _update_obs_noise(self):
        """
        Get observation noise (standard deviation).
        """
        return self._params[1]

dynamics = BurgersDynamics(dx, dt)

######################
# Fit model with GPR #
######################

kernel_1 = 1.0 * RBF(1.0)
kernel_2 = WhiteKernel()
kernel = kernel_1 + kernel_2
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, random_state=0).fit(obs_locs, obs_vals)
test_locs = [[x, t] for x in xx for t in tt]
gp_mean, gp_std = gp.predict(test_locs, return_std=True)
gp_mean = gp_mean.reshape(u.shape)
gp_std = gp_std.reshape(u.shape)
gp_ic = gp_mean[:,0]

vmin = u.min()
vmax = u.max()

plt.figure(figsize=(2,3))
plt.imshow(u, extent=(0, 2, -1, 1), origin="lower", aspect=2, rasterized=True, cmap="RdBu_r", vmin=vmin, vmax=vmax)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/burgers_eqn/burgers_gt2.pdf", transparent=True)

plt.figure(figsize=(2,3))
plt.imshow(gp_mean, extent=(0, 2, -1, 1), origin="lower", aspect=2, rasterized=True, cmap="RdBu_r", vmin=vmin, vmax=vmax)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.tight_layout(pad=0)
plt.savefig("figures/burgers_eqn/burgers_gpr.pdf", transparent=True)

##########################################
# Fit model with iterative linearisation #
##########################################

# Run model forward with initial condition from GPR
u_guess = odeint(burgers_odes, gp_ic, tt, args=(k, params[0],), mxstep=5000).T

# Fit model with iterative linearisation
max_iter = 1
model = IterativeRegressor(u, dynamics, u0=u_guess, mixing_coef=0.8)
model.fit(obs_dict, params, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)

############
# Plot fit #
############


fig, ax = plt.subplots(figsize=(2.5,3))
im = plt.imshow(model.posterior_mean, extent=(0, 2, -1, 1), origin="lower", aspect=2, rasterized=True, cmap="RdBu_r", vmin=vmin, vmax=vmax)
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.tight_layout(pad=0)
box = ax.get_position()
box.x0 = box.x0 - 0.05
box.x1 = box.x1 - 0.05
ax.set_position(box)
cax = fig.add_axes([ax.get_position().x1+0.08,ax.get_position().y0,0.05,ax.get_position().height])
cb_std = plt.colorbar(im, cax=cax)

plt.savefig("figures/burgers_eqn/burgers_spde.pdf", transparent=True)

# plt.figure(figsize=(3,3))
# plt.imshow(model.posterior_std, extent=(0, 2, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r")
# plt.xlabel("$t$")
# plt.ylabel("$x$", labelpad=0)
# plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
# plt.tight_layout(pad=0)
# plt.savefig("figures/burgers_eqn/burgers_spde_std.pdf", transparent=True)
plt.show()

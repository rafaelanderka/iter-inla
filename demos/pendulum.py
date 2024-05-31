import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.integrate import odeint
from findiff import FinDiff, Coef, Identity

from iinla import util
from iinla.nonlinear import SPDEDynamics, IterativeRegressor

# Set seed
np.random.seed(0)

#####################################################
# Generate data from nonlinear damped pendulum eqn. #
#####################################################

# Define parameters of damped pendulum and model
b = 0.3
c = 1.
obs_std = 1e-1
params = np.array([b, c, obs_std])

# Create temporal discretisation
L_t = 25                      # Duration of simulation [s]
dt = 0.1                      # Infinitesimal time
N_t = int(L_t / dt) + 1       # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array
T = np.around(T, decimals=1)

# Define the initial condition    
u0 = [0.75 * np.pi, 0.]

# Define corresponding system of ODEs
def pend(u, t, b, c):
    theta, omega = u
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

# Solve system of ODEs
u = odeint(pend, u0, T, args=(b, c,))

# For our purposes we only need the solution for the pendulum angle, theta
u = u[:, 0].reshape(1, -1)

# Sample observations
obs_std = 1e-1
obs_count = 20
obs_loc_1 = np.where(T == 5.)[0][0]
obs_dict = util.sample_observations(u, obs_count, obs_std, extent=(None, None, 0, obs_loc_1))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()))
print("Number of observations:", obs_idxs.shape[0])

######################################
# Define nonlinear pendulum dynamics #
######################################

class PendulumDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. b
    1. c
    2. observation noise
    """

    def __init__(self, dt) -> None:
        super().__init__()
        self.dt = dt

    def _update_diff_op(self):
        """
        Constructs linearised differential operator based on current state.
        """
        b = self._params[0]
        c = self._params[1]
        partial_t = FinDiff(1, self.dt, 1)
        partial_tt = FinDiff(1, self.dt, 2)
        u0_cos = np.cos(self._u0)
        diff_op = partial_tt + Coef(b) * partial_t + Coef(c * u0_cos) * Identity()
        return diff_op

    def _update_prior_precision(self):
        """
        Calculate current prior precision.
        """
        prior_precision = self._L.T @ self._L
        return prior_precision

    def _update_prior_mean(self):
        """
        Calculate current prior mean.
        """
        c = self._params[1]
        u0_cos = np.cos(self._u0)
        u0_sin = np.sin(self._u0)
        prior_mean = spsolve(self._L, (c * (self._u0 * u0_cos - u0_sin)).flatten())
        return prior_mean.reshape(self._u0.shape)

    def _update_obs_noise(self):
        """
        Get observation noise (standard deviation).
        """
        return self._params[2]


dynamics = PendulumDynamics(dt)

##########################################
# Fit model with iterative linearisation #
##########################################

max_iter = 20
model = IterativeRegressor(u, dynamics, mixing_coef=0.1)
model.fit(obs_dict, params, max_iter=max_iter, animated=False, calc_std=True, calc_mnll=True)

############
# Plot fit #
############

plt.plot(T, u.squeeze(), "b", label="Ground truth")
plt.plot(T, model.posterior_mean.squeeze(), "k", label="Posterior mean")
plt.plot(T, model.posterior_mean.squeeze() + model.posterior_std.squeeze(), "--", color="grey", label="Posterior std. dev.")
plt.plot(T, model.posterior_mean.squeeze() - model.posterior_std.squeeze(), "--", color="grey")
plt.axvline(dt * obs_loc_1, color='grey', ls=':')
plt.scatter(dt * obs_idxs[:,1], obs_vals, c="r", marker="x", label="Observations")
plt.legend(loc="best")
plt.xlabel("t")
plt.ylabel("$\\theta$")
plt.savefig("figures/pendulum/pendulum_sode.png", dpi=200)
plt.show()

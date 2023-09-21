import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from findiff import FinDiff, Coef, Identity

from spdeinf import nonlinear, util

# Load Allen-Cahn eq. data from PINNs examples
data = loadmat("data/PINNs/AC.mat")
uu = data['uu']
xx = data['x'].squeeze()
tt = data['tt'].squeeze()
N_x = xx.shape[0]
N_t = tt.shape[0]
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]

# Define Allen-Cahn eq. parameters
alpha = 0.001
beta = 5

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dx, dt, alpha, beta):
    """
    Constructs current linearised differential operator.
    """
    partial_t = FinDiff(1, dt, 1, acc=2)
    partial_xx = FinDiff(0, dx, 2, acc=2, periodic=True)
    u0_sq = u0 ** 2
    diff_op = partial_t - Coef(alpha) * partial_xx + Coef(3 * beta * u0_sq) * Identity() - Coef(beta) * Identity()
    return diff_op

def get_prior_mean(u0, diff_op_gen, beta):
    """
    Calculates current prior mean.
    """
    u0_cu = u0 ** 3
    diff_op = diff_op_gen(u0)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (2 * beta * u0_cu).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u: get_diff_op(u, dx, dt, alpha, beta)
prior_mean_gen = lambda u: get_prior_mean(u, diff_op_gen, beta)


######################################
# Naive diff. op. and mean generator #
######################################

def get_diff_op_naive(u0, dx, dt, alpha):
    """
    Constructs current linearised differential operator.
    """
    partial_t = FinDiff(1, dt, 1)
    partial_xx = FinDiff(0, dx, 2, periodic=True)
    diff_op = partial_t - Coef(alpha) * partial_xx
    return diff_op

def get_prior_mean_naive(u0, diff_op_gen, beta):
    u0_cu = u0 ** 3
    diff_op = diff_op_gen(u0)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (beta * (u0 - u0_cu)).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen_naive = lambda u: get_diff_op_naive(u, dx, dt, alpha)
prior_mean_gen_naive = lambda u: get_prior_mean_naive(u, diff_op_gen, beta)

## Fit GP with non-linear SPDE prior from Allen-Cahn equation

# Sample observations
obs_std = 1e-2
obs_count = 256
obs_dict = util.sample_observations(uu, obs_count, obs_std, extent=(None, None, 0, 56))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
print("Number of observations:", obs_idxs.shape[0])

# Perform iterative optimisation
max_iter = 50
model = nonlinear.NonlinearSPDERegressor(uu, dx, dt, diff_op_gen, prior_mean_gen, mixing_coef=0.5)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)
iter_count = len(model.mse_hist)

# Plot convergence history
plt.plot(np.arange(1, iter_count + 1), model.mse_hist, label="Linearisation via expansion")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.xticks(np.arange(2, iter_count + 1, 2))
plt.legend()
plt.savefig("figures/allen_cahn_eqn/mse_conv.png", dpi=200)
plt.show()

# Save animation
print("Saving animation...")
model.save_animation("figures/allen_cahn_eqn/allen_cahn_eqn_iter_animation.gif", fps=5)

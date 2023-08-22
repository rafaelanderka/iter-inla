import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from findiff import FinDiff, Coef, Identity

from spdeinf import nonlinear, util

# Load Korteweg-de Vries eq. data from PINNs examples
data = loadmat("data/PINNs/KdV.mat")
uu = data['uu'][::4,::4]
xx = data['x'].squeeze()[::4]
tt = data['tt'].squeeze()[::4]
N_x = xx.shape[0]
N_t = tt.shape[0]
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]
plt.imshow(uu, origin="lower")
plt.show()
print(tt)

# Define Korteweg-de Vries eq. parameters
l1 = 1
l2 = 0.0025

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dx, dt, l1, l2):
    """
    Constructs current linearised differential operator.
    """
    partial_t = FinDiff(1, dt, 1, acc=4)
    partial_x = FinDiff(0, dx, 1, acc=4)
    partial_xxx = FinDiff(0, dx, 3, acc=4)
    u0_x = partial_x(u0)
    diff_op = partial_t + Coef(l1 * u0) * partial_x + Coef(l1 * u0_x) * Identity() + Coef(l2) * partial_xxx
    return diff_op

def get_prior_mean(u0, diff_op_gen, l1):
    """
    Calculates current prior mean.
    """
    partial_x = FinDiff(0, dx, 1, acc=4)
    u0_x = partial_x(u0)
    diff_op = diff_op_gen(u0)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (l1 * u0 * u0_x).flatten())
    return prior_mean.reshape(u0.shape)

diff_op_gen = lambda u: get_diff_op(u, dx, dt, l1, l2)
prior_mean_gen = lambda u: get_prior_mean(u, diff_op_gen, l1)


## Fit GP with non-linear SPDE prior from Korteweg-de Vries equation

# Sample observations
obs_std = 1e-3
obs_count_1 = 100
obs_count_2 = 100
obs_loc_1 = np.where(tt == 0.2)[0][0]
obs_loc_2 = np.where(tt == 0.8)[0][0]
obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_1+1))
obs_dict.update(util.sample_observations(uu, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
print("Number of observations:", obs_idxs.shape[0])

# Perform iterative optimisation
max_iter = 10
model = nonlinear.NonlinearSPDERegressor(uu, dx, dt, diff_op_gen, prior_mean_gen, mixing_coef=1.)
model.fit(obs_dict, obs_std, max_iter=max_iter, animated=True, calc_std=True)
iter_count = len(model.mse_hist)

# Plot convergence history
plt.plot(np.arange(1, iter_count + 1), model.mse_hist, label="Linearisation via expansion")
# plt.plot(np.arange(1, iter_count + 1), model_naive.mse_hist, label="Naive linearisation")
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

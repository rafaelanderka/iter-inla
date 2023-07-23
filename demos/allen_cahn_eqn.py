import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from findiff import FinDiff, Coef, Identity
from sksparse.cholmod import cholesky

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
alpha = 0.0001
beta = 5

################################
# Diff. op. and mean generator #
################################

def get_diff_op(u0, dx, dt, alpha, beta):
    """
    Constructs current linearised differential operator.
    """
    partial_t = FinDiff(1, dt, 1)
    partial_xx = FinDiff(0, dx, 2)
    u0_sq = u0 ** 2
    diff_op = partial_t - Coef(alpha) * partial_xx + Coef(3 * beta * u0_sq) * Identity() - Coef(beta) * Identity()
    return diff_op

diff_op_gen = lambda u: get_diff_op(u, dx, dt, alpha, beta)
prior_mean_gen = lambda u: get_prior_mean(u, diff_op_gen, beta)

def get_prior_mean(u0, diff_op_gen, beta):
    """
    Calculates current prior mean.
    """
    u0_cu = u0 ** 3
    diff_op = diff_op_gen(u0)
    diff_op_mat = diff_op.matrix(u0.shape)
    prior_mean = spsolve(diff_op_mat, (2 * beta * u0_cu).flatten())
    return prior_mean.reshape(u0.shape)

## Fit GP with non-linear PDE prior from Burgers' equation

# Sample observations
obs_noise = 1e-4
obs_count = 256
obs_dict = util.sample_observations(uu, obs_count, obs_noise, xlim=20)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
print("Number of observations:", obs_idxs.shape[0])

# Perform iterative optimisation
max_iter = 20
model = nonlinear.NonlinearSPDERegressor(uu, dx, dt, diff_op_gen, prior_mean_gen)
model.fit(obs_dict, obs_noise, max_iter=max_iter, animated=True, calc_std=True)

# Check prior covariance
diff_op_init = diff_op_gen(np.zeros_like(uu))
diff_op_final = diff_op_gen(model.posterior_mean)

L_init = util.operator_to_matrix(diff_op_init, uu.shape, interior_only=False)
L_final = util.operator_to_matrix(diff_op_final, uu.shape, interior_only=False)
LL_init = L_init.T @ L_init
LL_final = L_final.T @ L_final
LL_init_chol = cholesky(LL_init + identity(LL_init.shape[0]))
LL_final_chol = cholesky(LL_final + identity(LL_final.shape[0]))
prior_init_std = np.sqrt(LL_init_chol.spinv().diagonal().reshape(uu.shape))
prior_final_std = np.sqrt(LL_final_chol.spinv().diagonal().reshape(uu.shape))

fig, ax = plt.subplots(1, 2)
im_init = ax[0].imshow(prior_init_std, origin="lower")
im_final = ax[1].imshow(prior_final_std, origin="lower")
ax[0].set_xlabel('x')
ax[0].set_ylabel('t')
fig.colorbar(im_init)
fig.colorbar(im_final)
plt.show()

# Save animation
print("Saving animation...")
model.save_animation("figures/allen_cahn_eqn/allen_cahn_eqn_iter_animation.gif", fps=3)

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from findiff import FinDiff, Coef, Identity

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from iinla import util
from iinla.nonlinear import SPDEDynamics, IterativeRegressor

# Set seed
np.random.seed(6)
data_id = 4
# 0:0, 1:1, 2:3, 3:5, 4:6

#########################################
# Load data from Korteweg-de Vries eqn. #
#########################################

# Define parameters of the Korteweg-de Vries eqn. and model
l1 = 1
l2 = 0.0025
obs_std = 1e-3
params = np.array([l1, l2, obs_std])

## Generate datasets
# Load Korteweg-de Vries eqn. data from PINNs examples
data = loadmat("data/PINNs/KdV.mat")
uu_full = data['uu']
xx_full = data['x'].squeeze()
tt_full = data['tt'].squeeze()
dx_full = xx_full[1] - xx_full[0]
dt_full = tt_full[1] - tt_full[0]
uu = uu_full[::4,::4]
xx = xx_full[::4]
tt = tt_full[::4]
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]
N_x = xx.shape[0]
N_t = tt.shape[0]
shape = (N_x, N_t)
Xgrid, Tgrid = np.meshgrid(xx, tt, indexing='ij')

# Sample observations
obs_count_1 = 20
obs_count_2 = 20
obs_loc_1 = np.where(tt == 0.2)[0][0]
obs_loc_2 = np.where(tt == 0.8)[0][0]
obs_dict = util.sample_observations(uu, obs_count_1, obs_std, extent=(None, None, obs_loc_1, obs_loc_1+1))
obs_dict.update(util.sample_observations(uu, obs_count_2, obs_std, extent=(None, None, obs_loc_2, obs_loc_2+1)))
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
obs_vals = np.array(list(obs_dict.values()), dtype=float)
obs_locs = np.array([[xx[i], tt[j]] for i, j in obs_idxs])
print("Number of observations:", obs_idxs.shape[0])

##########################################
# Define Korteweg-de Vries eqn. dynamics #
##########################################

class KdVDynamics(SPDEDynamics):
    """
    The parameters of the model are, in order:
    0. l1
    1. l2
    2. observation noise
    """

    def __init__(self, dx, dt) -> None:
        super().__init__()
        self.dx = dx
        self.dt = dt

    def _update_diff_op(self):
        """
        Construct current linearised differential operator.
        """
        l1 = self._params[0]
        l2 = self._params[1]
        partial_t = FinDiff(1, self.dt, 1, acc=2)
        partial_x = FinDiff(0, self.dx, 1, acc=2, periodic=True)
        partial_xxx = FinDiff(0, self.dx, 3, acc=2, periodic=True)
        u0_x = partial_x(self._u0)
        diff_op = partial_t + Coef(l1 * self._u0) * partial_x + Coef(l1 * u0_x) * Identity() + Coef(l2) * partial_xxx
        return diff_op

    def _update_prior_precision(self):
        """
        Calculate current prior precision.
        """
        # prior_precision = (self.dt * self.dx / self._params[2]**2) * (self._L.T @ self._L)
        prior_precision = self.dt * self.dx * (self._L.T @ self._L)
        # prior_precision = self._L.T @ self._L
        return prior_precision

    def _update_prior_mean(self):
        """
        Calculate current prior mean.
        """
        l1 = self._params[0]

        # Construct linearisation remainder term
        partial_x = FinDiff(0, self.dx, 1, acc=2, periodic=True)
        u0_x = partial_x(self._u0)
        remainder = l1 * self._u0 * u0_x

        # Compute prior mean
        prior_mean = spsolve(self._L, remainder.flatten())
        # prior_mean = spsolve(self._L.T @ self._L, self._L.T @ remainder.flatten())
        return prior_mean.reshape(self._u0.shape)

    def _update_obs_noise(self):
        """
        Get observation noise (standard deviation).
        """
        return self._params[2]

dynamics = KdVDynamics(dx, dt)

######################
# Fit model with GPR #
######################

# gp_kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=obs_std**2, noise_level_bounds="fixed")
# gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=10)
# gp.fit(obs_locs, obs_vals)
# test_locs = [[x, t] for x in xx for t in tt]
# test_locs_t0 = [[x, 0] for x in xx]
# ic_mean, ic_cov = gp.predict(test_locs_t0, return_cov=True)
# ic_std = np.sqrt(np.diag(ic_cov))

# # Save data for other benchmarks
# obs_table = np.empty((obs_count_1, 3))
# obs_table = [[xx[k[0]], tt[k[1]], v] for k, v in obs_dict.items()]
# util.obs_to_csv(obs_table, header="XTU", filename=f"data/KdVTrain{data_id}.csv")

# u_table = np.empty((N_x * N_t, 3))
# u_table[:,0] = Xgrid.flatten()
# u_table[:,1] = Tgrid.flatten()
# u_table[:,2] = uu.flatten()
# util.obs_to_csv(u_table, header="XTU", filename=f"data/KdVTest.csv")

# data_dict = {'uu': uu, 'uu_full': uu_full, 'xx': xx, 'xx_full': xx_full, 'tt': tt, 'tt_full': tt_full, 'dx': dx, 'dx_full': dx_full, 'dt': dt, 'dt_full': dt_full, 'u0_mean': ic_mean, 'u0_cov': ic_cov, 'u0_std': ic_std, 'obs_dict': obs_dict, 'obs_std': obs_std}
# data_file = f"data/kdv_{data_id}.pkl"
# with open(data_file, 'wb') as f:
#     pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

##########################################
# Fit model with iterative linearisation #
##########################################

max_iter = 20
model = IterativeRegressor(uu, dynamics, mixing_coef=0.1)
model.fit(obs_dict, params, max_iter=max_iter, animated=True, calc_std=True, calc_mnll=True)

############
# Plot fit #
############

plt.figure(figsize=(3,3))
im_mean = plt.imshow(model.posterior_mean, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="gnuplot2")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="white", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.colorbar(im_mean)
plt.tight_layout(pad=0)
plt.savefig("figures/kdv/kdv_spde.pdf", transparent=True)

plt.figure(figsize=(3,3))
im_std = plt.imshow(model.posterior_std, extent=(0, 1, -1, 1), origin="lower", aspect=.5, rasterized=True, cmap="YlGnBu_r")
plt.xlabel("$t$")
plt.ylabel("$x$", labelpad=0)
plt.scatter(dt * obs_idxs[:,1], dx * obs_idxs[:,0] - 1, c="grey", s=12, marker="o", alpha=1.0, label="Observations")
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
plt.colorbar(im_std)
plt.tight_layout(pad=0)
plt.savefig("figures/kdv/kdv_spde_std.pdf", transparent=True)
plt.show()

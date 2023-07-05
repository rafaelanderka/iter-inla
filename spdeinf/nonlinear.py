import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity
from sksparse.cholmod import cholesky

from . import linear
from . import metrics
from . import util

class NonlinearSPDERegressor(object):
    def __init__(self, u, dx, dt, diff_op_generator, mixing_coeff=1.) -> None:
        self.u = u
        self.dx = dx
        self.dt = dt
        self.dV = self.dx * self.dt
        self.diff_op_generator = diff_op_generator
        self.mixing_coeff = mixing_coeff
        self.persistance_coeff = 1. - self.mixing_coeff
        self.shape = self.u.shape

        # Data to fit
        self.obs_dict = None
        self.obs_noise = None

        # Optimiser state
        self.v = np.zeros_like(self.u)
        self.v_hist = [self.v.copy()]
        self.lml = float("-inf")
        self.preempt_requested = False
        self.sigma = 1000 

        # Posterior parameters
        self.posterior_mean = None
        self.posterior_std = None

    def update(self, calc_std=False, calc_lml=False):
        # Use current v to generate new approximate linear diff. operator
        diff_op_guess = self.diff_op_generator(self.v)

        # Construct precision matrix corresponding to the linear differential operator
        L = util.operator_to_matrix(diff_op_guess, self.shape, interior_only=False)
        LL = L.T @ L
        LL_chol = cholesky(LL + identity(LL.shape[0]))
        kappa = np.mean(LL_chol.spinv().diagonal())
        print(kappa)
        prior_precision = self.sigma * self.dV / kappa * LL

        # Fit corresponding GP
        res = linear._fit_gp(self.u, self.obs_dict, self.obs_noise, prior_precision, calc_std=calc_std, calc_lml=calc_lml)
        self.posterior_mean = res['posterior_mean']
        self.v = self.persistance_coeff * self.v + self.mixing_coeff * self.posterior_mean
        self.v_hist.append(self.v.copy())
        if calc_std:
            self.posterior_std = res['posterior_std']
        if calc_lml:
            self.lml = res['log_marginal_likelihood']

    def fit(self, obs_dict, obs_noise, max_iter, animated=False, calc_std=False, calc_lml=False):
        self.preempt_requested = False
        self.obs_dict = obs_dict
        self.obs_noise = obs_noise
        calc_std = calc_std or animated
        obs_idx = util.swap_cols(np.array(list(obs_dict.keys()), dtype=int))

        # Initialise figure
        if animated:
            gs_kw = dict(width_ratios=[1, 1, 1, 1], height_ratios=[1])
            fig, axd = plt.subplot_mosaic([['gt', 'mean', 'std', 'diff']], gridspec_kw=gs_kw, figsize=(17, 5))
            im_gt = axd['gt'].imshow(self.u.T, animated=True, origin="lower")
            im_mean = axd['mean'].imshow(np.zeros_like(self.u).T, animated=True, origin="lower")
            im_std = axd['std'].imshow(np.zeros_like(self.u).T, animated=True, origin="lower")
            im_diff = axd['diff'].imshow(np.zeros_like(self.u).T, animated=True, origin="lower")
            axd['mean'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
            axd['std'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
            axd['diff'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
            axd['gt'].set_title('Ground truth')
            axd['mean'].set_title('Posterior mean')
            axd['std'].set_title('Posterior std.')
            axd['diff'].set_title('$u - \mu_{u|y}$')
            fig.colorbar(im_gt, ax=axd['gt'])
            fig.colorbar(im_mean, ax=axd['mean'])
            fig.colorbar(im_std, ax=axd['std'])
            fig.colorbar(im_diff, ax=axd['diff'])
            fig.show()
            fig.canvas.mpl_connect('close_event', self.preempt)

        # Perform iterative linearisation
        print('Fitting model...')
        for i in range(max_iter):
            # Handle preempt requests
            if self.preempt_requested:
                break

            # Perform update
            is_final_iteration = i == max_iter - 1
            calc_std = calc_std or is_final_iteration
            calc_lml = calc_lml or is_final_iteration
            self.update(calc_std=calc_std, calc_lml=calc_lml)
            print(f'iter={i+1:d}, MSE={metrics.mse(self.u, self.v)}')

            # Draw and output the current parameters
            if animated:
                im_mean.set_data(self.posterior_mean.T)
                im_std.set_data(self.posterior_std.T)
                im_diff.set_data((self.u - self.posterior_mean).T)
                im_mean.autoscale()
                im_std.autoscale()
                im_diff.autoscale()
                fig.canvas.draw()
                fig.canvas.flush_events()
        return

    def preempt(self, *args):
        self.preempt_requested = True

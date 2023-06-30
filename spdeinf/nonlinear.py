import numpy as np
import matplotlib.pyplot as plt

from . import linear
from . import metrics
from . import util

class NonlinearSPDERegressor(object):
    def __init__(self, u, dx, dt, diff_op_generator, mixing_coeff=1.) -> None:
        self.u = u
        self.dx = dx
        self.dt = dt
        self.diff_op_generator = diff_op_generator
        self.mixing_coeff = mixing_coeff
        self.persistance_coeff = 1. - mixing_coeff

        # Data
        self.obs_dict = None
        self.obs_noise = None

        # Optimiser state
        self.v = np.zeros_like(self.u)
        self.v_hist = [self.v.copy()]
        self.log_marginal_likelihood = None
        self.preempt_requested = False

        # Posterior parameters
        self.posterior_mean = None
        self.posterior_std = None

    def fit(self, obs_dict, obs_noise, max_iter):
        self.preempt_requested = False
        self.obs_dict = obs_dict
        self.obs_noise = obs_noise
        obs_idx = util.swap_cols(np.array(list(obs_dict.keys()), dtype=int))

        # Initialise figure
        gs_kw = dict(width_ratios=[1, 1, 1], height_ratios=[1])
        fig, axd = plt.subplot_mosaic([['gt', 'mean', 'std']], gridspec_kw=gs_kw, figsize=(17, 5))
        im_gt = axd['gt'].imshow(self.u.T, animated=True, origin="lower")
        im_mean = axd['mean'].imshow(np.zeros_like(self.u).T, animated=True, origin="lower")
        im_std = axd['std'].imshow(np.zeros_like(self.u).T, animated=True, origin="lower")
        axd['mean'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['std'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['gt'].set_title('Ground truth')
        axd['mean'].set_title('Posterior mean')
        axd['std'].set_title('Posterior std.')
        fig.colorbar(im_mean, ax=axd['mean'])
        fig.colorbar(im_std, ax=axd['std'])
        fig.show()
        fig.canvas.mpl_connect('close_event', self.preempt)

        # Perform iterative linearisation
        print('Fitting model...')
        for i in range(max_iter):
            # Handle preempt requests
            if self.preempt_requested:
                break

            # Perform update
            self.update(calc_std=True)
            print(f'iter={i+1:d}, MSE={metrics.mse(self.u, self.v)}')

            # Draw and output the current parameters
            im_mean.set_data(self.posterior_mean.T)
            im_std.set_data(self.posterior_std.T)
            im_mean.autoscale()
            im_std.autoscale()
            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.close()
        return

    def update(self, calc_std=False):
        # Use current v to generate new approximate linear diff. operator
        diff_op_guess = self.diff_op_generator(self.v)
        res = linear.fit_spde_gp(self.u, self.obs_dict, self.obs_noise, diff_op_guess, calc_std=calc_std)
        self.posterior_mean = res['posterior_mean']
        self.v = self.persistance_coeff * self.v + self.mixing_coeff * self.posterior_mean
        self.v_hist.append(self.v.copy())
        if calc_std:
            self.posterior_std = res['posterior_std']

    def preempt(self, *args):
        self.preempt_requested = True

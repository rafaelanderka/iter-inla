import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.sparse import identity
from sksparse.cholmod import cholesky

from . import linear
from . import metrics
from . import util

class NonlinearSPDERegressor(object):
    def __init__(self, u, dx, dt, diff_op_generator, prior_mean_generator, mixing_coeff=1.) -> None:
        self.u = u
        self.dx = dx
        self.dt = dt
        self.dV = self.dx * self.dt
        self.diff_op_generator = diff_op_generator
        self.prior_mean_generator = prior_mean_generator
        self.mixing_coeff = mixing_coeff
        self.persistance_coeff = 1. - self.mixing_coeff
        self.shape = self.u.shape

        # Data to fit
        self.obs_dict = None
        self.obs_noise = None

        # Optimiser state
        self.u0 = np.zeros_like(self.u)
        # self.u0 = self.u.copy()
        self.u0_hist = [self.u0.copy()]
        self.mse = float("inf")
        self.mse_hist = []
        self.rmse = float("inf")
        self.rmse_hist = []
        self.lml = float("-inf")
        self.lml_hist = []
        self.preempt_requested = False
        self.sigma = 1000 

        # Prior/posterior parameters
        self.prior_mean = None
        self.prior_mean_hist = []
        self.data_term = None
        self.data_term_hist = []
        self.posterior_mean = None
        self.posterior_mean_hist = []
        self.posterior_std = None
        self.posterior_std_hist = []

    def update(self, calc_std=False, calc_lml=False, tol=1e-3):
        # Use current u0 to generate new approximate linear diff. operator
        diff_op_guess = self.diff_op_generator(self.u0)

        # Use current u0 to generate new prior mean
        self.prior_mean = self.prior_mean_generator(self.u0)
        self.prior_mean_hist.append(self.prior_mean)

        # Construct precision matrix corresponding to the linear differential operator
        L = util.operator_to_matrix(diff_op_guess, self.shape, interior_only=False)
        LL = L.T @ L
        LL_chol = cholesky(LL + tol * identity(LL.shape[0]))
        kappa = LL_chol.spinv().diagonal().mean()
        prior_precision = (self.sigma ** 2) / (self.dV * kappa) * LL

        # Subtract prior mean from observations
        obs_dict = self.obs_dict.copy()
        for idx in obs_dict.keys():
            obs_dict[idx] = obs_dict[idx] - self.prior_mean[idx]

        ## Fit corresponding GP
        # Get "data term" of posterior
        res = linear._fit_gp(self.u - self.prior_mean, obs_dict, self.obs_noise, prior_precision, calc_std=calc_std, calc_lml=calc_lml)
        self.data_term = res['posterior_mean']
        self.data_term_hist.append(self.data_term.copy())

        # Calculate full posterior mean as sum of prior mean and data term
        self.posterior_mean = self.prior_mean + self.data_term
        self.posterior_mean_hist.append(self.posterior_mean.copy())

        self.u0 = self.persistance_coeff * self.u0 + self.mixing_coeff * self.posterior_mean
        self.u0_hist.append(self.u0.copy())

        # Calculate MSE
        self.mse = metrics.mse(self.u, self.u0)
        self.mse_hist.append(self.mse)
        self.rmse = np.sqrt(self.mse)
        self.rmse_hist.append(self.rmse)

        # Optionally calculate std. dev. and log evidence
        if calc_std:
            self.posterior_std = res['posterior_std']
            self.posterior_std_hist.append(self.posterior_std.copy())
        if calc_lml:
            self.lml = res['log_marginal_likelihood']

    def fit(self, obs_dict, obs_noise, max_iter, animated=False, calc_std=False, calc_lml=False):
        self.preempt_requested = False
        self.obs_dict = obs_dict
        self.obs_noise = obs_noise
        calc_std = calc_std or animated

        # Initialise figure
        if animated:
            fig, im_mean, im_std, im_prior, im_data = self.init_animation()

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
            print(f'iter={i+1:d}, RMSE={self.rmse}')

            # Draw and output the current parameters
            if animated:
                self.update_animation(i, im_mean, im_std, im_prior, im_data)
                fig.canvas.draw()
                fig.canvas.flush_events()

        if animated:
            plt.close()

        return

    def init_animation(self):
        obs_idx = np.array(list(self.obs_dict.keys()), dtype=int)
        gs_kw = dict(width_ratios=[1, 1, 1, 1, 1], height_ratios=[1])
        fig, axd = plt.subplot_mosaic([['gt', 'mean', 'std', 'prior', 'data_term']], gridspec_kw=gs_kw, figsize=(17, 4))
        im_gt = axd['gt'].imshow(self.u, animated=True, origin="lower")
        im_mean = axd['mean'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        im_std = axd['std'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        # im_diff = axd['diff'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        im_prior = axd['prior'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        im_data = axd['data_term'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        axd['mean'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['std'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        # axd['diff'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['prior'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['data_term'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['gt'].set_title('Ground truth')
        axd['mean'].set_title('Posterior mean')
        axd['std'].set_title('Posterior std.')
        # axd['diff'].set_title('$u - \mu_{u|y}$')
        axd['prior'].set_title('Prior mean')
        axd['data_term'].set_title('Posterior mean\ndata term')
        fig.colorbar(im_gt, ax=axd['gt'])
        fig.colorbar(im_mean, ax=axd['mean'])
        fig.colorbar(im_std, ax=axd['std'])
        # fig.colorbar(im_diff, ax=axd['diff'])
        fig.colorbar(im_prior, ax=axd['prior'])
        fig.colorbar(im_data, ax=axd['data_term'])
        fig.tight_layout()
        fig.show()
        fig.canvas.mpl_connect('close_event', self.preempt)
        return fig, im_mean, im_std, im_prior, im_data

    def update_animation(self, i, im_mean, im_std, im_prior, im_data):
        im_mean.set_data(self.posterior_mean_hist[i])
        im_std.set_data(self.posterior_std_hist[i])
        # im_diff.set_data(self.u - self.posterior_mean_hist[i])
        im_prior.set_data(self.prior_mean_hist[i])
        im_data.set_data(self.data_term_hist[i])
        im_mean.autoscale()
        im_std.autoscale()
        # im_diff.autoscale()
        im_prior.autoscale()
        im_data.autoscale()
        return im_mean, im_std, im_prior, im_data

    def save_animation(self, output_filename, fps=5):
        fig, im_mean, im_std, im_prior, im_data = self.init_animation()
        animate = lambda i: self.update_animation(i, im_mean, im_std, im_prior, im_data)
        t_steps = len(self.posterior_mean_hist)  # Number of frames

        # Create an animation
        anim = animation.FuncAnimation(fig, animate, frames=t_steps, interval=10, blit=True)

        # Save the animation
        anim.save(output_filename, writer='pillow', fps=fps)

    def preempt(self, *args):
        self.preempt_requested = True

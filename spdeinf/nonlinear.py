import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import approx_fprime
from scipy.sparse import identity
from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError
from tqdm import tqdm

from . import inla, linear, metrics, util

class NonlinearSPDERegressor(object):
    def __init__(self, u, dx, dt, diff_op_generator, prior_mean_generator, mixing_coef=1.) -> None:
        self.u = u
        self.dx = dx
        self.dt = dt
        self.dV = self.dx * self.dt
        self.diff_op_generator = diff_op_generator
        self.prior_mean_generator = prior_mean_generator
        self.mixing_coef = mixing_coef
        self.persistance_coef = 1. - self.mixing_coef
        self.shape = self.u.shape

        # Data to fit
        self.obs_dict = None
        self.obs_std = None

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
        # LL_chol = cholesky(LL + tol * identity(LL.shape[0]))
        # kappa = LL_chol.spinv().diagonal().mean()
        # prior_precision = (self.sigma ** 2) / (self.dV * kappa) * LL
        # prior_precision = (self.sigma ** 2) / (self.dV) * LL
        prior_precision = LL

        ## Fit corresponding GP
        # Get "data term" of posterior
        try:
            res = linear._fit_gp(self.u, self.obs_dict, self.obs_std, self.prior_mean, prior_precision,
                                 calc_std=calc_std, calc_lml=calc_lml)
        except CholmodNotPositiveDefiniteError:
            print("Posterior precision positive definite")
            self.preempt_requested = True
            return

        # Store results
        self.posterior_mean = res['posterior_mean']
        self.posterior_mean_hist.append(self.posterior_mean.copy())
        self.data_term = res['posterior_mean_data_term']
        self.data_term_hist.append(self.data_term.copy())

        # Perform damped update
        self.u0 = self.persistance_coef * self.u0 + self.mixing_coef * self.posterior_mean
        self.u0_hist.append(self.u0.copy())

        # Calculate MSE
        self.mse = metrics.mse(self.u, self.u0)
        self.mse_hist.append(self.mse)
        self.rmse = np.sqrt(self.mse)
        self.rmse_hist.append(self.rmse)

        # Optionally store std. dev. and log evidence if it was computed
        if calc_std:
            self.posterior_std = res['posterior_std']
            self.posterior_std_hist.append(self.posterior_std.copy())
        if calc_lml:
            self.lml = res['log_marginal_likelihood']

    def fit(self, obs_dict, obs_std, max_iter, animated=False, calc_std=False, calc_lml=False):
        self.preempt_requested = False
        self.obs_dict = obs_dict
        self.obs_std = obs_std
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
            if animated and not self.preempt_requested:
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
        t_steps = len(self.mse_hist)  # Number of frames

        # Create an animation
        anim = animation.FuncAnimation(fig, animate, frames=t_steps, interval=10, blit=True)

        # Save the animation
        anim.save(output_filename, writer='pillow', fps=fps)

    def preempt(self, *args):
        self.preempt_requested = True


class NonlinearINLASPDERegressor(object):
    def __init__(self, u, dx, dt, param0, diff_op_generator, prior_mean_generator, logpdf_marginal_posterior,
                 mixing_coef=1., params_true=None, param_bounds=None, sampling_evec_scales=None) -> None:
        self.u = u
        self.dx = dx
        self.dt = dt
        self.dV = self.dx * self.dt
        self.param0 = param0
        self.diff_op_generator = diff_op_generator
        self.prior_mean_generator = prior_mean_generator
        self.logpdf_marginal_posterior = logpdf_marginal_posterior
        self.mixing_coef = mixing_coef
        self.persistance_coef = 1. - self.mixing_coef
        self.params_true = params_true
        self.param_bounds = param_bounds
        self.sampling_evec_scales = sampling_evec_scales
        self.shape = self.u.shape
        
        # Determine animation type
        if self.u.shape[0] == 1:
            self.plot_1d = True
        else:
            self.plot_1d = False

        # Data to fit
        self.obs_dict = None
        self.obs_idxs_flat = None
        self.obs_count = 0
        self.obs_std = None

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
        # self.prior_mean = None
        # self.prior_mean_hist = []
        # self.data_term = None
        # self.data_term_hist = []
        self.posterior_mean = None
        self.posterior_mean_hist = []
        self.posterior_std = None
        self.posterior_std_hist = []
        self.log_marg_post_hist = []
        self.samples_x_hist = []

    def update(self, calc_std=False, calc_lml=False, tol=1e-3):
        # Compute posterior marginals
        logpdf = lambda x, return_conditional_params=False: self.logpdf_marginal_posterior(x, self.u0, self.obs_dict, self.diff_op_generator, self.prior_mean_generator,
                                                                                           return_conditional_params=return_conditional_params)
        try:
            samples, H_v = inla.sample_parameter_posterior(logpdf, self.param0, param_bounds=self.param_bounds, sampling_evec_scales=self.sampling_evec_scales)
        except CholmodNotPositiveDefiniteError:
            print("Posterior precision not positive definite")
            self.preempt_requested = True
            return
        self.posterior_mean, self.posterior_std = inla.compute_field_posterior_stats(samples)
        self.posterior_mean_hist.append(self.posterior_mean.copy())
        self.posterior_std_hist.append(self.posterior_std.copy())
        self.samples_x_hist.append(samples[0])

        # Sweep marginal posterior for plotting
        nu_count = 4
        nus = np.linspace(*self.param_bounds[0], nu_count)

        t_obs_count = 4
        t_obss = np.linspace(*self.param_bounds[1], t_obs_count)

        log_marg_post = np.empty((nu_count, t_obs_count))
        for i, a in tqdm(enumerate(nus), total=nu_count):
            for j, t_obs in enumerate(t_obss):
                log_marg_post[i,j] = logpdf([a, t_obs])
        self.log_marg_post_hist.append((nus, t_obss, log_marg_post))

        # Perform damped update for next field estimate u0
        self.u0 = self.persistance_coef * self.u0 + self.mixing_coef * self.posterior_mean
        self.u0_hist.append(self.u0.copy())

        # Calculate MSE
        self.mse = metrics.mse(self.u, self.u0)
        self.mse_hist.append(self.mse)
        self.rmse = np.sqrt(self.mse)
        self.rmse_hist.append(self.rmse)

    def fit(self, obs_dict, obs_std, max_iter, animated=False, calc_std=False, calc_lml=False):
        self.preempt_requested = False
        self.obs_dict = obs_dict
        self.obs_count = len(obs_dict)
        self.obs_std = obs_std
        calc_std = calc_std or animated

        # Initialise figure
        if animated:
            fig, im_mean, im_std, ax_log_marg = self.init_animation()

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
            if animated and not self.preempt_requested:
                self.update_animation(i, im_mean, im_std, ax_log_marg)
                fig.canvas.draw()
                fig.canvas.flush_events()

        if animated:
            plt.close()

        return

    def init_animation(self):
        obs_idx = np.array(list(self.obs_dict.keys()), dtype=int)
        obs_val = np.array(list(self.obs_dict.values()))

        gs_kw = dict(width_ratios=[1, 1, 1, 1], height_ratios=[1])
        fig, axd = plt.subplot_mosaic([['gt', 'mean', 'std',  'log_marg']], gridspec_kw=gs_kw, figsize=(13, 4))
        if self.plot_1d:
            im_gt = axd['gt'].plot(self.u[0])
            im_mean = axd['mean'].plot(np.zeros_like(self.u[0]))[0]
            im_std = axd['std'].plot(np.zeros_like(self.u[0]))[0]
            axd['mean'].scatter(obs_idx[:,1], obs_val, c='r', marker='x')
            axd['std'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        else:
            im_gt = axd['gt'].imshow(self.u, animated=True, origin="lower")
            im_mean = axd['mean'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
            im_std = axd['std'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
            axd['mean'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
            axd['std'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
            fig.colorbar(im_gt, ax=axd['gt'])
            fig.colorbar(im_mean, ax=axd['mean'])
            fig.colorbar(im_std, ax=axd['std'])

        # Configure titles and labels
        axd['gt'].set_title('Ground truth')
        axd['mean'].set_title('Posterior mean')
        axd['std'].set_title('Posterior std.')
        axd['log_marg'].set_title('$\\log \\widetilde{p}(\\theta | y)$')
        axd['log_marg'].set_xlabel('$c$')
        axd['log_marg'].set_ylabel('$b$')

        fig.tight_layout()
        fig.show()
        fig.canvas.mpl_connect('close_event', self.preempt)
        return fig, im_mean, im_std, axd['log_marg']

    def update_animation(self, i, im_mean, im_std, ax_log_marg):
        if self.plot_1d:
            im_mean.set_ydata(self.u0_hist[i][0])
            im_std.set_ydata(self.posterior_std_hist[i][0])
            # im_mean.set_ydata(self.posterior_mean_hist[i][0])
        else:
            im_mean.set_data(self.u0_hist[i])
            im_std.set_data(self.posterior_std_hist[i])
            im_mean.autoscale()
            im_std.autoscale()

        ax_log_marg.clear()
        im_log_marg = ax_log_marg.contourf(self.log_marg_post_hist[i][1], self.log_marg_post_hist[i][0], self.log_marg_post_hist[i][2], levels=50)
        ax_log_marg.scatter(self.samples_x_hist[i][0,1], self.samples_x_hist[i][0,0], c='r', marker='x', label="MAP $\\theta$")
        # plt.scatter(t_obs_prior_mode, nu_prior_mode, c='b', marker='x', label="Prior mode $\\theta$")
        if self.params_true is not None:
            ax_log_marg.scatter(self.params_true[1], self.params_true[0], c='m', marker='x', label="True $\\theta$")
        ax_log_marg.scatter(self.samples_x_hist[i][:,1], self.samples_x_hist[i][:,0], s=5, c='k', label="Sampled points")
        # plt.quiver(*H_v_origins, H_v[1,:], H_v[0,:], width=0.005, scale=8, label="Eigenvectors of Hessian")
        ax_log_marg.set_xlabel('$c$')
        ax_log_marg.set_ylabel('$b$')
        ax_log_marg.set_title('$\\log \\widetilde{p}(\\theta | y)$')
        ax_log_marg.legend()
        return im_mean, im_std

    def save_animation(self, output_filename, fps=5):
        fig, im_mean, im_std, ax_log_marg = self.init_animation()
        animate = lambda i: self.update_animation(i, im_mean, im_std, ax_log_marg)
        t_steps = len(self.mse_hist)  # Number of frames

        # Create an animation
        anim = animation.FuncAnimation(fig, animate, frames=t_steps, interval=10, blit=True)

        # Save the animation
        anim.save(output_filename, writer='pillow', fps=fps)

    def preempt(self, *args):
        self.preempt_requested = True

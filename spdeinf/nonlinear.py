import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import stats
from scipy import sparse
from scipy.optimize import approx_fprime
from scipy.sparse import identity
from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List

from . import inla, linear, metrics, util, distributions

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
        self.mnll = float("-inf")
        self.mnll_hist = []
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

    def update(self, calc_std=False, calc_mnll=False, tol=1e-3):
        # Use current u0 to generate new approximate linear diff. operator
        diff_op_guess = self.diff_op_generator(self.u0)

        # Use current u0 to generate new prior mean
        self.prior_mean = self.prior_mean_generator(self.u0)
        self.prior_mean_hist.append(self.prior_mean)

        # Construct precision matrix corresponding to the linear differential operator
        L = util.operator_to_matrix(diff_op_guess, self.shape, interior_only=False)
        prior_precision = L.T @ L

        ## Fit corresponding GMRF
        # Get "data term" of posterior
        try:
            res = linear._fit_gmrf(self.u, self.obs_dict, self.obs_std, self.prior_mean, prior_precision,
                                 calc_std=calc_std, calc_mnll=calc_mnll)
        except CholmodNotPositiveDefiniteError:
            print("Posterior precision not positive definite")
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
        if calc_mnll:
            self.mnll = res['mnll']
            self.mnll_hist.append(self.mnll)

    def fit(self, obs_dict, obs_std, max_iter, animated=False, calc_std=False, calc_mnll=False):
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
            calc_mnll = calc_mnll or is_final_iteration
            self.update(calc_std=calc_std, calc_mnll=calc_mnll)
            if calc_mnll:
                print(f'iter={i+1:d}, RMSE={self.rmse}, MNLL={self.mnll}')
            else:
                print(f'iter={i+1:d}, RMSE={self.rmse}')

            # Draw and output the current parameters
            if animated and not self.preempt_requested:
                self.update_animation(i, fig, im_mean, im_std, im_prior, im_data)
                fig.canvas.draw()
                fig.canvas.flush_events()

        if animated:
            plt.close()

        return self.u0.copy(), self.posterior_mean.copy(), self.posterior_std.copy()

    def init_animation(self):
        obs_idx = np.array(list(self.obs_dict.keys()), dtype=int)
        gs_kw = dict(width_ratios=[1, 1, 1, 1, 1], height_ratios=[1])
        fig, axd = plt.subplot_mosaic([['gt', 'mean', 'std', 'prior', 'data_term']], gridspec_kw=gs_kw, figsize=(17, 4))
        im_gt = axd['gt'].imshow(self.u, animated=True, origin="lower")
        im_mean = axd['mean'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        im_std = axd['std'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        im_prior = axd['prior'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        im_data = axd['data_term'].imshow(np.zeros_like(self.u), animated=True, origin="lower")
        axd['mean'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['std'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['prior'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['data_term'].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
        axd['gt'].set_title('Ground truth')
        axd['mean'].set_title('Posterior mean')
        axd['std'].set_title('Posterior std.')
        axd['prior'].set_title('Prior mean')
        axd['data_term'].set_title('Posterior mean\ndata term')
        fig.colorbar(im_gt, ax=axd['gt'])
        fig.colorbar(im_mean, ax=axd['mean'])
        fig.colorbar(im_std, ax=axd['std'])
        fig.colorbar(im_prior, ax=axd['prior'])
        fig.colorbar(im_data, ax=axd['data_term'])
        fig.tight_layout()
        fig.show()
        fig.canvas.mpl_connect('close_event', self.preempt)
        return fig, im_mean, im_std, im_prior, im_data

    def update_animation(self, i, fig, im_mean, im_std, im_prior, im_data):
        fig.suptitle(f'Iteration = {i+1}', y=1, fontsize=12)
        im_mean.set_data(self.posterior_mean_hist[i])
        im_std.set_data(self.posterior_std_hist[i])
        im_prior.set_data(self.prior_mean_hist[i])
        im_data.set_data(self.data_term_hist[i])
        im_mean.autoscale()
        im_std.autoscale()
        im_prior.autoscale()
        im_data.autoscale()
        return im_mean, im_std, im_prior, im_data

    def save_animation(self, output_filename, fps=5):
        fig, im_mean, im_std, im_prior, im_data = self.init_animation()
        animate = lambda i: self.update_animation(i, fig, im_mean, im_std, im_prior, im_data)
        t_steps = len(self.mse_hist)  # Number of frames

        # Create an animation
        anim = animation.FuncAnimation(fig, animate, frames=t_steps, interval=10, blit=True)

        # Save the animation
        anim.save(output_filename, writer='pillow', fps=fps)

    def preempt(self, *args):
        self.preempt_requested = True


class NonlinearINLASPDERegressor(object):
    def __init__(self, u, dx, dt, param0, diff_op_generator, prior_mean_generator, logpdf_marginal_posterior,
                 mixing_coef=1., params_true=None, param_bounds=None, sampling_evec_scales=None, sampling_threshold=None) -> None:
        self.u = u
        self.dx = dx
        self.dt = dt
        self.dV = self.dx * self.dt
        self.param0 = param0
        # self.plot_param_post = False
        self.plot_param_post = param0.shape[0] == 2
        self.diff_op_generator = diff_op_generator
        self.prior_mean_generator = prior_mean_generator
        self.logpdf_marginal_posterior = logpdf_marginal_posterior
        self.mixing_coef = mixing_coef
        self.persistance_coef = 1. - self.mixing_coef
        self.params_true = params_true
        self.param_bounds = param_bounds
        self.sampling_evec_scales = sampling_evec_scales
        self.sampling_threshold = sampling_threshold
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
        self.mnll = float("-inf")
        self.mnll_hist = []
        self.preempt_requested = False
        self.sigma = 1000 
        self.params_opt = None

        # Conditional/posterior parameters
        self.posterior_mean = None
        self.posterior_mean_hist = []
        self.posterior_std = None
        self.posterior_std_hist = []
        self.log_marg_post_hist = []
        self.samples_x_hist = []

    def update(self, calc_std=False, calc_mnll=False, tol=1e-3):
        # Compute posterior marginals
        logpdf = lambda x, return_conditional_params=False: self.logpdf_marginal_posterior(x, self.u0, self.obs_dict, self.diff_op_generator, self.prior_mean_generator,
                                                                                           return_conditional_params=return_conditional_params)
        try:
            samples, H_v, params_opt, p_uy = inla.sample_parameter_posterior(logpdf, self.param0, param_bounds=self.param_bounds, sampling_evec_scales=self.sampling_evec_scales, sampling_threshold=self.sampling_threshold)
        except CholmodNotPositiveDefiniteError:
            print("Posterior precision not positive definite")
            self.preempt_requested = True
            return
        self.posterior_mean, self.posterior_std = inla.compute_field_posterior_stats(samples)
        self.posterior_mean_hist.append(self.posterior_mean.copy())
        self.posterior_std_hist.append(self.posterior_std.copy())
        self.samples_x_hist.append(samples[0])
        self.params_opt = params_opt

        # Calculate MSE
        self.mse = metrics.mse(self.u, self.u0)
        self.mse_hist.append(self.mse)
        self.rmse = np.sqrt(self.mse)
        self.rmse_hist.append(self.rmse)

        # Compute negative log predictive likelihood 
        if calc_mnll:
            self.mnll = -stats.norm.logpdf(self.u.flatten(), loc=self.posterior_mean.flatten(), scale=self.posterior_std.flatten()).mean()
            self.mnll_hist.append(self.mnll)

        # Sweep marginal posterior for plotting
        if self.plot_param_post:
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

    def fit(self, obs_dict, obs_std, max_iter, animated=False, calc_std=False, calc_mnll=False):
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
            self.update(calc_std=calc_std, calc_mnll=calc_mnll)
            if calc_mnll:
                print(f'iter={i+1:d}, RMSE={self.rmse}, MNLL={self.mnll}')
            else:
                print(f'iter={i+1:d}, RMSE={self.rmse}')

            # Draw and output the current parameters
            if animated and not self.preempt_requested:
                self.update_animation(i, fig, im_mean, im_std, ax_log_marg)
                fig.canvas.draw()
                fig.canvas.flush_events()

        if animated:
            plt.close()

        return self.u0.copy(), self.posterior_mean.copy(), self.posterior_std.copy()

    def init_animation(self):
        obs_idx = np.array(list(self.obs_dict.keys()), dtype=int)
        obs_val = np.array(list(self.obs_dict.values()), dtype=float)

        subfig_labels = ['gt', 'mean', 'std']
        subfig_widths = [1, 1, 1]
        if self.plot_param_post:
            subfig_labels += ['log_marg']
            subfig_widths += [1]
        gs_kw = dict(width_ratios=subfig_widths, height_ratios=[1])
        fig, axd = plt.subplot_mosaic([subfig_labels], gridspec_kw=gs_kw, figsize=(11, 4))
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
        if self.plot_param_post:
            axd['log_marg'].set_title('$\\log \\widetilde{p}(\\theta | y)$')
            axd['log_marg'].set_xlabel('$c$')
            axd['log_marg'].set_ylabel('$b$')

        fig.tight_layout()
        fig.show()
        fig.canvas.mpl_connect('close_event', self.preempt)
        if self.plot_param_post:
            return fig, im_mean, im_std, axd['log_marg']
        return fig, im_mean, im_std, None

    def update_animation(self, i, fig, im_mean, im_std, ax_log_marg):
        fig.suptitle(f'Iteration = {i+1}', y=1, fontsize=12)
        if self.plot_param_post:
            ax_log_marg.clear()
            im_log_marg = ax_log_marg.contourf(self.log_marg_post_hist[i][1], self.log_marg_post_hist[i][0], self.log_marg_post_hist[i][2], levels=50)
            if self.params_true is not None:
                ax_log_marg.scatter(self.params_true[1], self.params_true[0], c='m', marker='x', label="True $\\theta$")
            ax_log_marg.scatter(self.param0[1], self.param0[0], c='b', marker='x', label="Prior mode $\\theta$")
            ax_log_marg.scatter(self.samples_x_hist[i][0,1], self.samples_x_hist[i][0,0], c='r', marker='x', label="MAP $\\theta$")
            ax_log_marg.scatter(self.samples_x_hist[i][:,1], self.samples_x_hist[i][:,0], s=5, c='k', label="Sampled points")
            ax_log_marg.set_xlabel('$c$')
            ax_log_marg.set_ylabel('$b$')
            ax_log_marg.set_title('$\\log \\widetilde{p}(\\theta | y)$')
            ax_log_marg.legend()

        if self.plot_1d:
            im_mean.set_ydata(self.u0_hist[i][0])
            im_std.set_ydata(self.posterior_std_hist[i][0])
        else:
            im_mean.set_data(self.u0_hist[i])
            im_std.set_data(self.posterior_std_hist[i])
            im_mean.autoscale()
            im_std.autoscale()

        return im_mean, im_std

    def save_animation(self, output_filename, fps=5):
        fig, im_mean, im_std, ax_log_marg = self.init_animation()
        animate = lambda i: self.update_animation(i, fig, im_mean, im_std, ax_log_marg)
        t_steps = len(self.mse_hist)  # Number of frames

        # Create an animation
        anim = animation.FuncAnimation(fig, animate, frames=t_steps, interval=10, blit=True)

        # Save the animation
        anim.save(output_filename, writer='pillow', fps=fps)

    def preempt(self, *args):
        self.preempt_requested = True


####################################################
# ST: Changes below to adapt to the new update rule
####################################################

class AbstractNonlinearINLASPDERegressor(ABC):
    def __init__(self, u, dx, dt,
                 param0,
                 param_priors: List[distributions.Distribution] = None,
                 param_bounds=None,
                 params_true=None,
                 mixing_coef=1.,
                 sampling_evec_scales=None,
                 sampling_threshold=None) -> None:
        """
        Parent class for a Nonlinear INLA-SPDE Regressor.
        When subclassing, remember to override the following methods:

        1. _get_diff_op()
        2. _get_prior_mean()
        3. _get_prior_precision()
        4. _get_obs_noise()

        Parameters
        ----------
        param0: array
            Initial guess of the parameters to optimise the log marginal likelihood
        param_priors: list of Distribution
            Prior distributions imposed on the parameters specified in param0
        param_bounds: list
            Bounds on the parameters when optimising
        params_true: bool

        mixing_coef: float
            Amount of damping to be applied
        sampling_evec_scales:

        sampling_threshold:

        """
        assert len(param0) == len(param_priors), "param0 and param_priors must have the same length"
        assert len(param0) == len(param_bounds), "param0 and param_bounds must have the same length"
        assert len(param0) == len(sampling_evec_scales), "param0 and sampling_evec_scales must have the same length"

        self.u = u
        self.dx = dx
        self.dt = dt
        self.dV = self.dx * self.dt
        self.param0 = param0
        self.param_priors = param_priors
        # self.plot_param_post = False
        self.plot_param_post = param0.shape[0] == 2
        self.mixing_coef = mixing_coef
        self.persistance_coef = 1. - self.mixing_coef
        self.params_true = params_true
        self.param_bounds = param_bounds
        self.sampling_evec_scales = sampling_evec_scales
        self.sampling_threshold = sampling_threshold
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
        self.mnll = float("-inf")
        self.mnll_hist = []
        self.preempt_requested = False
        self.sigma = 1000 
        self.params_opt = None
        self.marginal_dist_u_y = None # to store marginal Gaussian mixture model p(u_i | y)

        # Conditional/posterior parameters
        self.posterior_mean = None
        self.posterior_mean_hist = []
        self.posterior_std = None
        self.posterior_std_hist = []
        self.log_marg_post_hist = []
        self.samples_x_hist = []

    @abstractmethod
    def _get_diff_op(self, u0, params, *args, **kwargs):
        """
        Constructs current linearised differential operator.
        """
        return NotImplementedError

    @abstractmethod
    def _get_prior_precision(self, u0, params, **kwargs):
        """
        Calculates current prior precision.
        """
        return NotImplementedError

    @abstractmethod
    def _get_prior_mean(self, u0, params, **kwargs):
        """
        Calculates current prior mean.
        """
        return NotImplementedError

    @abstractmethod
    def _get_obs_noise(self, params, **kwargs):
        """
        Get the observation noise (standard deviation).
        """
        return self.obs_std # By default, return self.obs_std if observation noise is fixed. Customise if learned.

    def _log_param_prior(self, params):
        """
        Compute the log prior on the parameters Σ_i log(p(θ_i)).
        """
        log_p_t = 0.0
        for p, prior_dist in zip(params, self.param_priors):
            log_p_t += prior_dist.logpdf(p)
        return log_p_t

    def _log_state_prior(self, mu_u, mu_uy, Q_u, Q_u_logdet):
        """
        Compute the log prior on the model state log(p(u|θ)).
        """
        diff_mu_uy_mu_u = mu_uy - mu_u
        log_p_ut = 0.5 * (Q_u_logdet - diff_mu_uy_mu_u.T @ Q_u @ diff_mu_uy_mu_u - self.M * self.log_2pi)
        return log_p_ut

    def _log_likelihood(self, obs_vals, obs_idxs_flat, mu_uy, Q_obs, Q_obs_logdet):
        """
        Compute the log likelihood log(p(y|u,θ)).
        """
        diff_obs_mu_uy = obs_vals - mu_uy[obs_idxs_flat]
        log_p_yut = 0.5 * (Q_obs_logdet - diff_obs_mu_uy.T @ Q_obs @ diff_obs_mu_uy - self.N * self.log_2pi)
        return log_p_yut

    def _logpdf_marginal_posterior(self, params, Q_u, Q_uy, Q_obs, mu_u, mu_uy, obs_dict, shape, regularisation=1e-3, **kwargs):
        """
        Compute the log marginal on parameters.
        log(p(θ|y)) = log(p(θ)) + log(p(u|θ)) + log(p(y|u,θ)) - log(p(u|y,θ)) + const.
        """

        # Define reused consts.
        self.log_2pi = np.log(2 * np.pi)
        self.N = Q_obs.shape[0]
        self.M = Q_u.shape[0]

        # Unpack parameters
        obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
        obs_idxs_flat = shape[1] * obs_idxs[:,0] + obs_idxs[:,1]
        obs_vals = np.array(list(obs_dict.values()), dtype=float)

        # Perform matrix factorisation to compute log determinants
        Q_u_chol = cholesky(Q_u + regularisation * sparse.identity(Q_u.shape[0]))
        Q_u_logdet = Q_u_chol.logdet()
        Q_uy_chol = cholesky(Q_uy)
        Q_uy_logdet = Q_uy_chol.logdet()
        Q_obs_chol = cholesky(Q_obs)
        Q_obs_logdet = Q_obs_chol.logdet()

        # Compute approximate log posterior log(p(θ|y))
        log_p_t = self._log_param_prior(params)
        log_p_ut = self._log_state_prior(mu_u, mu_uy, Q_u, Q_u_logdet)
        log_p_yut = self._log_likelihood(obs_vals, obs_idxs_flat, mu_uy, Q_obs, Q_obs_logdet)
        log_p_uyt = 0.5 * (Q_uy_logdet - self.M * self.log_2pi)
        log_p_ty = log_p_t + log_p_ut + log_p_yut - log_p_uyt
        return log_p_ty

    def logpdf_marginal_posterior(self, params, u0, obs_dict, parameterisation='moment', return_conditional_params=False, debug=False, **kwargs):
        """
        Return the approximate log marginal log(p(θ|y)) and optionally, the marginal statistics on the states if return_conditional_params=True
        """
        
        # Process args
        obs_count = len(obs_dict.keys())
        obs_noise = self._get_obs_noise(params, **kwargs)

        # Compute prior mean and precision
        prior_mean = self._get_prior_mean(u0, params, **kwargs)
        prior_precision = self._get_prior_precision(u0, params, **kwargs)

        # Get "data term" of full conditional
        res = linear._fit_gmrf(self.u, obs_dict, obs_noise, prior_mean, prior_precision, calc_std=return_conditional_params,
                            include_initial_cond=False, return_posterior_precision=True, return_posterior_shift=True, regularisation=1e-5)
        
        # Define prior and full condtional params
        mu_u = prior_mean
        Q_u = prior_precision
        mu_uy = res['posterior_mean']
        Q_uy = res['posterior_precision']
        Q_obs = sparse.diags([obs_noise ** (-2)], 0, shape=(obs_count, obs_count), format='csc')

        if debug:
            plt.figure()
            plt.plot(prior_mean[0])
            plt.show()

        # Compute marginal posterior
        logpdf = self._logpdf_marginal_posterior(params, Q_u, Q_uy, Q_obs, mu_u.flatten(), mu_uy.flatten(), obs_dict, u0.shape, **kwargs)
        if return_conditional_params:
            # if parameterisation == 'moment':
            #     return logpdf, mu_uy, res['posterior_var']
            # elif parameterisation == 'natural':
            #     return logpdf, res['posterior_shift'], Q_uy
            # else:
            #     NotImplementedError
            return logpdf, mu_uy, res['posterior_var'], res['posterior_shift'], Q_uy
        return logpdf

    def update(self, calc_std=False, calc_mnll=False, parameterisation='natural', **kwargs):
        """
        Update the linearisation point in the iteration.

        Parameters
        ----------
        calc_std: bool
            Option to compute the state uncertainty at each iteration.
        calc_mnll: bool
            Option to compute and display the mean nll at each iteration.
        parameterisation: 'moment' or 'natural'
            Whether to apply update rule based on the moment parameterisation or the natural parameterisation.
        kwargs:
            Keyword arguments to be passed to the abstract methods.

        Note
        ----
        - When parameterisation='moment', we use the averaged posterior mean to update the linearisation point.
        - When parameterisation='natural', we use the averaged natural parameters to update the linearisation point.
          Specifically, we take averages of the shift vector b and the precision matrix P, and compute the corresonding
          mean by m = P^{-1}b, which will be used as the linearisation point.

        """
        # Compute posterior marginals
        logpdf = lambda x, return_conditional_params=False: self.logpdf_marginal_posterior(x, self.u0, self.obs_dict,
                                                                                           parameterisation=parameterisation,
                                                                                           return_conditional_params=return_conditional_params,
                                                                                           **kwargs)
        try:
            samples, H_v, params_opt, p_uy = inla.sample_parameter_posterior(logpdf, self.param0,
                                                                             param_bounds=self.param_bounds,
                                                                             sampling_evec_scales=self.sampling_evec_scales,
                                                                             sampling_threshold=self.sampling_threshold
                                                                            )
            self.marginal_dist_u_y = p_uy

        except CholmodNotPositiveDefiniteError:
            print("Posterior precision not positive definite")
            self.preempt_requested = True
            return

        out = inla.compute_field_posterior_stats(samples, parameterisation, calc_std)

        if calc_std:
            self.posterior_mean, self.posterior_std = out
            self.posterior_mean = self.posterior_mean.reshape(self.u.shape)
            self.posterior_std = self.posterior_std.reshape(self.u.shape)
            self.posterior_mean_hist.append(self.posterior_mean.copy())
            self.posterior_std_hist.append(self.posterior_std.copy())
        else:
            self.posterior_mean = out
            self.posterior_mean = self.posterior_mean.reshape(self.u.shape)
            self.posterior_mean_hist.append(self.posterior_mean.copy())

        self.samples_x_hist.append(samples[0])
        self.params_opt = params_opt

        # Calculate MSE
        self.mse = metrics.mse(self.u, self.u0)
        self.mse_hist.append(self.mse)
        self.rmse = np.sqrt(self.mse)
        self.rmse_hist.append(self.rmse)

        # Compute negative log predictive likelihood 
        if calc_mnll:
            # self.mnll = -stats.norm.logpdf(self.u.flatten(), loc=self.posterior_mean.flatten(), scale=self.posterior_std.flatten()).mean()
            self.mnll = -p_uy.logpdf(self.u).mean()
            self.mnll_hist.append(self.mnll)

        # Sweep marginal posterior for plotting
        if self.plot_param_post:
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

    def fit(self, obs_dict, obs_std, max_iter, parameterisation='natural', animated=False, calc_std=False, calc_mnll=False, **kwargs):
        """
        Fit nonlinear SPDE model on data.

        Parameters
        ----------
        obs_dict: dict
            Dictionary of observations.
        obs_std: float or None
            Standard deviation of noise if available.
        max_iter: int
            Maximum number of iterations
        parameterisation: 'moment' or 'natural'
            Whether to apply update rule based on the moment parameterisation or the natural parameterisation.
        animated: bool
            Option to display performance progress with animation.
        calc_std: bool
            Option to compute the state uncertainty at each iteration.
        calc_mnll: bool
            Option to compute and display the mean nll at each iteration.
        kwargs:
            Keyword arguments to be passed to the abstract methods.

        Note
        ----
        - When parameterisation='moment', we use the averaged posterior mean to update the linearisation point.
        - When parameterisation='natural', we use the averaged natural parameters to update the linearisation point.
          Specifically, we take averages of the shift vector b and the precision matrix P, and compute the corresonding
          mean by m = P^{-1}b, which will be used as the linearisation point.

        """
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
            self.update(calc_std=calc_std, calc_mnll=calc_mnll, parameterisation=parameterisation, **kwargs)
            if calc_mnll:
                print(f'iter={i+1:d}, RMSE={self.rmse}, MNLL={self.mnll}')
            else:
                print(f'iter={i+1:d}, RMSE={self.rmse}')

            # Draw and output the current parameters
            if animated and not self.preempt_requested:
                self.update_animation(i, fig, im_mean, im_std, ax_log_marg)
                fig.canvas.draw()
                fig.canvas.flush_events()

        if animated:
            plt.close()

        return self.u0.copy(), self.posterior_mean.copy(), self.posterior_std.copy()

    ##### Extra features to produce animation and plots of iterations #####
    def init_animation(self):
        obs_idx = np.array(list(self.obs_dict.keys()), dtype=int)
        obs_val = np.array(list(self.obs_dict.values()), dtype=float)

        subfig_labels = ['gt', 'mean', 'std']
        subfig_widths = [1, 1, 1]
        if self.plot_param_post:
            subfig_labels += ['log_marg']
            subfig_widths += [1]
        gs_kw = dict(width_ratios=subfig_widths, height_ratios=[1])
        fig, axd = plt.subplot_mosaic([subfig_labels], gridspec_kw=gs_kw, figsize=(11, 4))
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
        if self.plot_param_post:
            axd['log_marg'].set_title('$\\log \\widetilde{p}(\\theta | y)$')
            axd['log_marg'].set_xlabel('$c$')
            axd['log_marg'].set_ylabel('$b$')

        fig.tight_layout()
        fig.show()
        fig.canvas.mpl_connect('close_event', self.preempt)
        if self.plot_param_post:
            return fig, im_mean, im_std, axd['log_marg']
        return fig, im_mean, im_std, None

    def update_animation(self, i, fig, im_mean, im_std, ax_log_marg):
        fig.suptitle(f'Iteration = {i+1}', y=1, fontsize=12)
        if self.plot_param_post:
            ax_log_marg.clear()
            im_log_marg = ax_log_marg.contourf(self.log_marg_post_hist[i][1], self.log_marg_post_hist[i][0], self.log_marg_post_hist[i][2], levels=50)
            if self.params_true is not None:
                ax_log_marg.scatter(self.params_true[1], self.params_true[0], c='m', marker='x', label="True $\\theta$")
            ax_log_marg.scatter(self.param0[1], self.param0[0], c='b', marker='x', label="Prior mode $\\theta$")
            ax_log_marg.scatter(self.samples_x_hist[i][0,1], self.samples_x_hist[i][0,0], c='r', marker='x', label="MAP $\\theta$")
            ax_log_marg.scatter(self.samples_x_hist[i][:,1], self.samples_x_hist[i][:,0], s=5, c='k', label="Sampled points")
            ax_log_marg.set_xlabel('$c$')
            ax_log_marg.set_ylabel('$b$')
            ax_log_marg.set_title('$\\log \\widetilde{p}(\\theta | y)$')
            ax_log_marg.legend()

        if self.plot_1d:
            im_mean.set_ydata(self.u0_hist[i][0])
            im_std.set_ydata(self.posterior_std_hist[i][0])
        else:
            im_mean.set_data(self.u0_hist[i])
            im_std.set_data(self.posterior_std_hist[i])
            im_mean.autoscale()
            im_std.autoscale()

        return im_mean, im_std

    def save_animation(self, output_filename, fps=5):
        fig, im_mean, im_std, ax_log_marg = self.init_animation()
        animate = lambda i: self.update_animation(i, fig, im_mean, im_std, ax_log_marg)
        t_steps = len(self.mse_hist)  # Number of frames

        # Create an animation
        anim = animation.FuncAnimation(fig, animate, frames=t_steps, interval=10, blit=True)

        # Save the animation
        anim.save(output_filename, writer='pillow', fps=fps)

    def preempt(self, *args):
        self.preempt_requested = True

import numbers
import numpy as np
from sksparse.cholmod import cholesky
from scipy import sparse, stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from . import util

def fit_spde_gmrf(u, obs_dict, obs_std, diff_op, prior_mean=0, c=1, calc_std=False, calc_mnll=False,
                include_initial_cond=False, include_terminal_cond=False, include_boundary_cond=False,
                regularisation=0, return_prior_precision=False, return_posterior_precision=False):
    # Construct precision matrix corresponding to the linear differential operator
    mat = util.operator_to_matrix(diff_op, u.shape, interior_only=False)
    prior_precision = c * (mat.T @ mat)

    return _fit_gmrf(u, obs_dict, obs_std, prior_mean, prior_precision, calc_std=calc_std, calc_mnll=calc_mnll,
                    include_initial_cond=include_initial_cond, include_terminal_cond=include_terminal_cond,
                    include_boundary_cond=include_boundary_cond, regularisation=regularisation,
                    return_prior_precision=return_prior_precision, return_posterior_precision=return_posterior_precision)

def _fit_gmrf(ground_truth, obs_dict, obs_std, prior_mean, prior_precision, calc_std=False, calc_mnll=False,
            include_initial_cond=False, include_terminal_cond=False, include_boundary_cond=False,
            regularisation=0, return_prior_precision=False, return_posterior_precision=False, return_posterior_shift=False):
    # Process args
    shape = ground_truth.shape
    N = np.prod(shape)
    gt_prior_diff = (ground_truth - prior_mean).flatten()
    obs_idxs = list(obs_dict.keys())
    obs_precision = obs_std**(-2)

    # Get boundary indices
    boundary_idxs = util.get_boundary_indices(shape, include_initial_cond=include_initial_cond, 
                                              include_terminal_cond=include_terminal_cond,
                                              include_boundary_cond=include_boundary_cond)

    # Construct observation mask and posterior shift
    grid_idxs = util.get_domain_indices(shape)
    mask = np.zeros(N)
    posterior_shift_data_term = np.zeros(np.prod(shape))
    for idx in obs_idxs:
        pr_m = prior_mean if isinstance(prior_mean, numbers.Number) else prior_mean[idx]
        mask[grid_idxs[tuple(idx)]] = obs_precision
        posterior_shift_data_term[grid_idxs[idx]] = (obs_dict[idx] - pr_m) * obs_precision

    obs_mask = mask.copy().astype(bool)
    for idx in boundary_idxs:
        mask[idx] = obs_precision
        posterior_shift_data_term[idx] = gt_prior_diff[idx] * obs_precision

    # Construct posterior precision
    posterior_precision = prior_precision + sparse.diags(mask, format="csr")
    if regularisation != 0:
        posterior_precision += regularisation * sparse.identity(N)

    # Compute posterior mean
    res = dict()
    posterior_precision_cholesky = cholesky(posterior_precision)
    posterior_mean_data_term = posterior_precision_cholesky(posterior_shift_data_term).reshape(shape)
    posterior_mean = prior_mean + posterior_mean_data_term
    res['posterior_mean'] = posterior_mean
    res['posterior_mean_data_term'] = posterior_mean_data_term

    # Optionally compute posterior variance/std.
    if calc_std or calc_mnll:
        posterior_var = posterior_precision_cholesky.spinv().diagonal().reshape(shape)
        posterior_std = np.sqrt(posterior_var)
        res['posterior_var'] = posterior_var
        res['posterior_std'] = posterior_std

    # Optionally compute negative log predictive likelihood 
    if calc_mnll:
        residuals = ground_truth - posterior_mean
        mnll = -stats.norm.logpdf(ground_truth.flatten(), loc=posterior_mean.flatten(), scale=posterior_std.flatten()).mean()
        res['mnll'] = mnll

    # Optionally return prior precision
    if return_prior_precision:
        res['prior_precision'] = prior_precision

    # Optionally return posterior precision
    if return_posterior_precision:
        res['posterior_precision'] = posterior_precision
        res['posterior_precision_chol'] = posterior_precision_cholesky

    # Optionally return posterior shift
    if return_posterior_shift:
        if isinstance(prior_mean, numbers.Number):
            posterior_shift = posterior_precision * prior_mean + posterior_shift_data_term
        else:
            posterior_shift = posterior_precision @ prior_mean.flatten() + posterior_shift_data_term
        res['posterior_shift'] = posterior_shift

    return res

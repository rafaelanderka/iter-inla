import numbers
import numpy as np
from sksparse.cholmod import cholesky
from scipy import sparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from . import util

def fit_spde_gp(u, obs_dict, obs_std, diff_op, prior_mean=0, c=1, calc_std=False, calc_lml=False,
                include_initial_cond=False, include_terminal_cond=False, include_boundary_cond=False,
                regularisation=0, return_prior_precision=False, return_posterior_precision=False):
    # Construct precision matrix corresponding to the linear differential operator
    mat = util.operator_to_matrix(diff_op, u.shape, interior_only=False)
    prior_precision = c * (mat.T @ mat)

    return _fit_gp(u, obs_dict, obs_std, prior_mean, prior_precision, calc_std=calc_std, calc_lml=calc_lml,
                    include_initial_cond=include_initial_cond, include_terminal_cond=include_terminal_cond,
                    include_boundary_cond=include_boundary_cond, regularisation=regularisation,
                    return_prior_precision=return_prior_precision, return_posterior_precision=return_posterior_precision)

def _fit_gp(ground_truth, obs_dict, obs_std, prior_mean, prior_precision, calc_std=False, calc_lml=False,
            include_initial_cond=False, include_terminal_cond=False, include_boundary_cond=False,
            regularisation=0, return_prior_precision=False, return_posterior_precision=False):
    # Process args
    shape = ground_truth.shape
    N = np.prod(shape)
    gt_prior_diff = (ground_truth - prior_mean).flatten()
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    obs_precision = obs_std**(-2)

    # Get boundary indices
    boundary_idxs = util.get_boundary_indices(shape, include_initial_cond=include_initial_cond, 
                                              include_terminal_cond=include_terminal_cond,
                                              include_boundary_cond=include_boundary_cond)

    # Construct observation mask
    grid_idxs = util.get_domain_indices(shape)
    mask = np.zeros(N)
    for idx in obs_idxs:
        mask[grid_idxs[tuple(idx)]] = obs_precision
    obs_mask = mask.copy().astype(bool)
    for idx in boundary_idxs:
        mask[idx] = obs_precision

    # Construct posterior precision
    posterior_precision = prior_precision + sparse.diags(mask, format="csr")
    if regularisation != 0:
        posterior_precision += regularisation * sparse.identity(N)

    # Construct posterior shift
    posterior_shift = np.zeros(np.prod(shape))
    for idx in obs_idxs:
        idx = tuple(idx)
        pr_m = prior_mean if isinstance(prior_mean, numbers.Number) else prior_mean[idx]
        posterior_shift[grid_idxs[idx]] = (obs_dict[idx] - pr_m) * obs_precision
    for idx in boundary_idxs:
        posterior_shift[idx] = gt_prior_diff[idx] * obs_precision
    
    # Compute posterior mean
    res = dict()
    posterior_precision_cholesky = cholesky(posterior_precision)
    posterior_mean_data_term = posterior_precision_cholesky(posterior_shift).reshape(shape)
    res['posterior_mean'] = prior_mean + posterior_mean_data_term
    res['posterior_mean_data_term'] = posterior_mean_data_term

    # Optionally compute posterior variance/std.
    if calc_std:
        posterior_var = posterior_precision_cholesky.spinv().diagonal().reshape(shape)
        res['posterior_var'] = posterior_var
        res['posterior_std'] = np.sqrt(posterior_var)

    # Optionally compute log evidence
    if calc_lml:
        P = posterior_precision[obs_mask, :][:, obs_mask]
        L = cholesky(P)
        y = np.array([obs_dict[tuple(idx)] for idx in obs_idxs])
        n = len(y)
        res['log_marginal_likelihood'] = 0.5 * (L.logdet() - y.T @ P @ y - n * np.log(2 * np.pi))

    # Optionally return prior precision
    if return_prior_precision:
        res['prior_precision'] = prior_precision

    # Optionally return posterior precision
    if return_posterior_precision:
        res['posterior_precision'] = posterior_precision
        res['posterior_precision_chol'] = posterior_precision_cholesky
    return res

def fit_rbf_gp(u, obs_dict, X_test, dx, dt, obs_std):
    shape = u.shape
    # Extract the observations from the dictionary
    obs_idx = np.array(list(obs_dict.keys()), dtype=float)
    obs_idx[:,0] *= dx
    obs_idx[:,1] *= dt
    obs_vals = list(obs_dict.values())

    # # Add boundary conditions to obs_idx and obs_vals
    # boundary_indices = []
    # boundary_values = []

    # for i in range(shape[0]):
    #     for j in range(shape[1]):
    #         if j == 0 or j == shape[1] - 1:
    #             idx = np.array([i * dx, j * dt])
    #             val = u[i, j]
    #             boundary_indices.append(idx)
    #             boundary_values.append(val)

    # obs_idx = np.vstack((obs_idx, boundary_indices))
    # obs_vals = np.hstack((obs_vals, boundary_values))

    # Define the RBF kernel with hyperparameters l and sigma_f
    # l = 2  # Length scale
    # kernel = RBF(length_scale=l, length_scale_bounds='fixed')
    kernel = RBF()

    # Create a Gaussian Process Regressor with the RBF kernel
    gp = GaussianProcessRegressor(kernel=kernel, alpha=obs_std)

    # Train the GP model on the observations
    gp.fit(obs_idx, obs_vals)

    # Make predictions with the GP model over the input grid
    y_pred, std_dev = gp.predict(X_test, return_std=True)
    y_pred = y_pred.reshape(shape)
    std_dev = std_dev.reshape(shape)
    return y_pred, std_dev

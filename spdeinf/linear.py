import numpy as np
from sksparse.cholmod import cholesky
from scipy import sparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from . import util

def fit_spde_gp(u, obs_dict, obs_noise, diff_op, calc_std=False, calc_lml=False,
                 include_initial_cond=False, include_terminal_cond=False, include_boundary_cond=True):
    # Construct precision matrix corresponding to the linear differential operator
    mat = util.operator_to_matrix(diff_op, u.shape, interior_only=False)
    prior_precision = mat.T @ mat

    return _fit_gp(u, obs_dict, obs_noise, prior_precision, calc_std=calc_std, calc_lml=calc_lml,
                    include_initial_cond=include_initial_cond, include_terminal_cond=include_terminal_cond,
                    include_boundary_cond=include_boundary_cond)

def _fit_gp(ground_truth, obs_dict, obs_noise, prior_precision, calc_std=False, calc_lml=False,
             include_initial_cond=False, include_terminal_cond=False, include_boundary_cond=True):
    # Process args
    shape = ground_truth.shape
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    obs_noise_inv_sq = obs_noise**(-2)

    # Construct posterior precision and posterior shift
    N = np.prod(shape)
    grid_idxs = util.get_domain_indices(shape)
    boundary_idxs = util.get_boundary_indices(shape, include_initial_cond=include_initial_cond, 
                                              include_terminal_cond=include_terminal_cond,
                                              include_boundary_cond=include_boundary_cond)
    mask = np.zeros(N)
    for idx in obs_idxs:
        mask[grid_idxs[tuple(idx)]] = obs_noise_inv_sq
    obs_mask = mask.copy().astype(bool)
    for idx in boundary_idxs:
        mask[idx] = obs_noise_inv_sq
    posterior_precision = prior_precision + sparse.diags(mask, format="csr")
    posterior_shift = np.zeros(np.prod(shape))
    for idx in obs_idxs:
        posterior_shift[grid_idxs[tuple(idx)]] = obs_dict[tuple(idx)] * obs_noise_inv_sq
    for idx in boundary_idxs:
        posterior_shift[idx] = ground_truth.flatten()[idx]/obs_noise**2
    
    # Compute posterior mean and covariance
    res = dict()
    posterior_precision_cholesky = cholesky(posterior_precision)
    res['posterior_mean'] = posterior_precision_cholesky(posterior_shift).reshape(shape)
    if calc_std:
        posterior_var = posterior_precision_cholesky.spinv().diagonal().reshape(shape)
        res['posterior_std'] = np.sqrt(posterior_var)

    # Compute log marginal likelihood
    if calc_lml:
        P = posterior_precision[obs_mask, :][:, obs_mask]
        L = cholesky(P)
        y = np.array([obs_dict[tuple(idx)] for idx in obs_idxs])
        n = len(y)
        res['log_marginal_likelihood'] = 0.5 * (L.logdet() - y.T @ P @ y - n * np.log(2 * np.pi))
    return res

def fit_rbf_gp(u, obs_dict, X_test, dx, dt, obs_noise):
    shape = u.shape
    # Extract the observations from the dictionary
    obs_idx = np.array(list(obs_dict.keys()), dtype=float)
    obs_idx[:,0] *= dt
    obs_idx[:,1] *= dx
    obs_vals = list(obs_dict.values())

    # Add boundary conditions to obs_idx and obs_vals
    boundary_indices = []
    boundary_values = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            if j == 0 or j == shape[1] - 1:
                idx = np.array([i * dt, j * dx])
                val = u[i, j]
                boundary_indices.append(idx)
                boundary_values.append(val)

    obs_idx = np.vstack((obs_idx, boundary_indices))
    obs_vals = np.hstack((obs_vals, boundary_values))

    # Define the RBF kernel with hyperparameters l and sigma_f
    l = 0.2  # Length scale
    kernel = RBF(length_scale=l, length_scale_bounds='fixed')

    # Create a Gaussian Process Regressor with the RBF kernel
    gp = GaussianProcessRegressor(kernel=kernel, alpha=obs_noise)

    # Train the GP model on the observations
    gp.fit(util.swap_cols(obs_idx), obs_vals)

    # Make predictions with the GP model over the input grid
    y_pred, std_dev = gp.predict(X_test, return_std=True)
    y_pred = y_pred.reshape(shape)
    std_dev = std_dev.reshape(shape)
    return y_pred, std_dev

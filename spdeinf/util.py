import csv
import numpy as np

def operator_to_matrix(diff_op, shape, interior_only=True):
    """
    Convert a findiff operator into a precision matrix
    """
    mat = diff_op.matrix(shape)
    # mat = apply_boundary_conditions(mat, shape)
    if interior_only:
        interior_idxs = get_interior_indices(shape)
        mat = mat[interior_idxs]
        mat = mat[:, interior_idxs]
    return mat

def apply_boundary_conditions(mat, shape):
    """
    Apply non-zero boundary conditions to the precision matrix
    """
    n_rows, n_cols = shape
    for i in range(n_rows):
        for j in range(n_cols):
            if i == 0 or i == n_rows - 1 or j == 0 or j == n_cols - 1:
                idx = i * n_cols + j
                mat[idx, :] = 0
                mat[:, idx] = 0
                mat[idx, idx] = 1
    return mat

def get_domain_indices(shape):
    """
    Get grid indices
    """
    siz = np.prod(shape)
    full_indices = np.array(list(range(siz))).reshape(shape)
    return full_indices

def get_interior_indices(shape):
    """
    Get indices for domain interior
    """
    full_indices = get_domain_indices(shape)
    interior_slice = tuple(slice(1,-1) for _ in range(len(shape)))
    interior_indices = full_indices[interior_slice].flatten()
    return interior_indices

def get_exterior_indices(shape):
    """
    Get indices for domain exterior
    """
    full_indices = get_domain_indices(shape)
    mask = np.ones(shape, dtype=bool)
    mask[len(shape) * (slice(1, -1),)] = False
    exterior_indices = full_indices[mask].flatten()
    return exterior_indices

def get_boundary_indices(shape, include_initial_cond=False, include_terminal_cond=False, include_boundary_cond=True):
    """
    Get indices for domain boundary
    """
    # Parse args to determine which indices to keep
    row_start = 0
    row_end = None
    col_start = 0
    col_end = None
    if include_initial_cond:
        col_start = 1
    if include_terminal_cond:
        col_end = -1
    if include_boundary_cond:
        row_start = 1
        row_end = -1

    # Genarate boundary indices from mask
    full_indices = get_domain_indices(shape)
    mask = np.ones(shape, dtype=bool)
    mask[(slice(row_start, row_end),) + (len(shape) - 1) * (slice(col_start, col_end),)] = False
    boundary_indices = full_indices[mask].flatten()
    return boundary_indices

def sample_observations(u, obs_count, obs_noise, extent=(None, None, None, None), seed=None):
    """
    Sample noisy observations from field u at random locations
    """
    grid_size, time_size = u.shape
    x_idxs = np.arange(grid_size)[extent[0]:extent[1]]
    t_idxs = np.arange(time_size)[extent[2]:extent[3]]
    X_idxs, T_idxs = np.meshgrid(x_idxs, t_idxs, indexing='ij')
    all_idxs = np.stack([X_idxs.flatten(), T_idxs.flatten()], axis=1)
    if seed is None:
        idxs = np.random.choice(len(all_idxs), obs_count, replace=False)
        idxs = all_idxs[idxs]
    else:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(all_idxs, obs_count, replace=False)
    obs_dict = {tuple(idx): u[tuple(idx)]+obs_noise*np.random.randn() for idx in idxs}
    return obs_dict

def swap_cols(arr, i=0, j=1):
    """
    Swap columns i and j in arr (copy)
    """
    new = arr.copy()
    new.T[[i, j]] = new.T[[j, i]]
    return new

def obs_to_csv(tuples, header=None, filename='output.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        if header:
            writer.writerow(header)
        
        for tup in tuples:
            writer.writerow(tup)


# Functions to compute credible intervals (taken from the package "pypfilt")

def qtl_wt(x, weights, probs):
    """
    Calculate weighted quantiles of an array of values, where each value has a
    fractional weighting.

    Weights are summed over exact ties, yielding distinct values x_1 < x_2 <
    ... < x_N, with corresponding weights w_1, w_2, ..., w_N.
    Let ``s_j`` denote the sum of the first j weights, and let ``W`` denote
    the sum of all the weights.
    For a probability ``p``:

    - If ``p * W < s_1`` the estimated quantile is ``x_1``.
    - If ``s_j < p * W < s_{j + 1}`` the estimated quantile is ``x_{j + 1}``.
    - If ``p * W == s_N`` the estimated quantile is ``x_N``.
    - If ``p * W == s_j`` the estimated quantile is ``(x_j + x_{j + 1}) / 2``.

    :param x: A 1-D array of values.
    :param weights: A 1-D array of weights.
    :param probs: The quantile(s) to compute.

    :return: The array of weighted quantiles.

    :raises ValueError: if ``x`` or ``weights`` are not one-dimensional, or if
        ``x`` and ``weights`` have different dimensions.
    """

    if len(x.shape) != 1:
        message = 'x is not 1D and has shape {}'
        raise ValueError(message.format(x.shape))
    if len(weights.shape) != 1:
        message = 'weights is not 1D and has shape {}'
        raise ValueError(message.format(weights.shape))
    if x.shape != weights.shape:
        message = 'x and weights have different shapes {} and {}'
        raise ValueError(message.format(x.shape, weights.shape))

    # Remove values with zero or negative weights.
    # Weights of zero can arise if a particle is deemed sufficiently unlikely
    # given the recent observations and resampling has not (yet) occurred.
    if any(weights <= 0):
        mask = weights > 0
        weights = weights[mask]
        x = x[mask]

    # Sort x and the weights.
    i = np.argsort(x)
    x = x[i]
    weights = weights[i]

    # Combine duplicated values into a single sample.
    if any(np.diff(x) == 0):
        unique_xs = np.unique(x)
        weights = [sum(weights[v == x]) for v in unique_xs]
        x = unique_xs

    nx = len(x)
    cum_weights = np.cumsum(weights)
    net_weight = np.sum(weights)
    eval_cdf_locns = np.array(probs) * net_weight

    # Decide how strictly to compare probabilities to cumulative weights.
    atol = 1e-10

    # Define the bisection of lower and upper indices.
    def bisect(ix_lower, ix_upper):
        """
        Bisect the interval spanned by lower and upper indices.
        Returns ``None`` when the interval cannot be further divided.
        """
        if ix_upper > ix_lower + 1:
            return np.rint((ix_lower + ix_upper) / 2).astype(int)
        else:
            return None

    # Evaluate each quantile in turn.
    quantiles = np.zeros(len(probs))
    for locn_ix, locn in enumerate(eval_cdf_locns):
        # Check whether the quantile is the very first or last value.
        if cum_weights[0] >= (locn - atol):
            # NOTE: use strict equality with an absolute tolerance.
            if np.abs(locn - cum_weights[0]) <= atol and nx > 1:
                # Average over the two matching values.
                quantiles[locn_ix] = 0.5 * (x[0] + x[1])
            else:
                quantiles[locn_ix] = x[0]
            continue
        if cum_weights[-1] <= (locn - atol):
            quantiles[locn_ix] = x[-1]
            continue

        # Search the entire range of values.
        ix_lower = 0
        ix_upper = nx - 1

        # Find the smallest index in cum_weights that is greater than or equal
        # to the location at which to evaluate the CDF.
        ix_mid = bisect(ix_lower, ix_upper)
        while ix_mid is not None:
            w_mid = cum_weights[ix_mid]

            # NOTE: use strict equality with an absolute tolerance.
            if w_mid >= (locn - atol):
                ix_upper = ix_mid
            else:
                ix_lower = ix_mid
            ix_mid = bisect(ix_lower, ix_upper)

        # NOTE: use strict equality with an absolute tolerance.
        if np.abs(locn - cum_weights[ix_upper]) <= atol and ix_upper < nx - 1:
            # Average over the two matching values.
            quantiles[locn_ix] = 0.5 * (x[ix_upper] + x[ix_upper + 1])
        else:
            quantiles[locn_ix] = x[ix_upper]

    return quantiles


def cred_wt(x, weights, creds):
    """Calculate weighted credible intervals.

    :param x: A 1-D array of values.
    :param weights: A 1-D array of weights.
    :param creds: The credible interval(s) to compute (``0..100``, where ``0``
        represents the median and ``100`` the entire range).
    :type creds: List(int)

    :return: A dictionary that maps credible intervals to the lower and upper
        interval bounds.

    :raises ValueError: if ``x`` or ``weights`` are not one-dimensional, or if
        ``x`` and ``weights`` have different dimensions.
    """
    if len(x.shape) != 1:
        message = 'x is not 1D and has shape {}'
        raise ValueError(message.format(x.shape))
    if len(weights.shape) != 1:
        message = 'weights is not 1D and has shape {}'
        raise ValueError(message.format(weights.shape))
    if x.shape != weights.shape:
        message = 'x and weights have different shapes {} and {}'
        raise ValueError(message.format(x.shape, weights.shape))

    creds = sorted(creds)
    median = creds[0] == 0
    if median:
        creds = creds[1:]
    probs = [[0.5 - cred / 200.0, 0.5 + cred / 200.0] for cred in creds]
    probs = [pr for pr_list in probs for pr in pr_list]
    if median:
        probs = [0.5] + probs
    qtls = qtl_wt(x, weights, probs)
    intervals = {}
    if median:
        intervals[0] = (qtls[0], qtls[0])
        qtls = qtls[1:]
    for cred in creds:
        intervals[cred] = (qtls[0], qtls[1])
        qtls = qtls[2:]
    return intervals
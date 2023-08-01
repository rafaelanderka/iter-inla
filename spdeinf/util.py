import numpy as np
from jax.experimental import sparse as jsparse

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

def sample_observations(u, obs_count, obs_noise, extent=(None, None, None, None), seed=0):
    """
    Sample noisy observations from field u at random locations
    """
    grid_size, time_size = u.shape
    rng = np.random.default_rng(seed)
    x_idxs = np.arange(grid_size)[extent[0]:extent[1]]
    t_idxs = np.arange(time_size)[extent[2]:extent[3]]
    X_idxs, T_idxs = np.meshgrid(x_idxs, t_idxs, indexing='ij')
    all_idxs = np.stack([X_idxs.flatten(), T_idxs.flatten()], axis=1)
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

def scipy2jax_csr(A):
    """
    Convert a SciPy CSR to a Jax CSR matrix
    """
    return jsparse.CSR((A.data, A.indices, A.indptr), shape=A.shape)

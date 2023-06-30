from time import time
import numpy as np
from sksparse.cholmod import cholesky, cholesky_AAt
from scipy import sparse
from jax.experimental import sparse as jsparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.core.display import HTML

def mse(x1, x2):
    return np.mean((x1 - x2) ** 2)

def swap_cols(arr, i=0, j=1):
    new = arr.copy()
    new.T[[i, j]] = new.T[[j, i]]
    return new

def scipy2jax_csr(A):
    return jsparse.CSR((A.data, A.indices, A.indptr), shape=A.shape)

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
    
def get_boundary_indices(shape):
    """
    Get indices for domain boundary
    """
    full_indices = get_domain_indices(shape)
    mask = np.ones(shape, dtype=bool)
    mask[(slice(0, None),) + (len(shape) - 1) * (slice(1, -1),)] = False
    # mask[:,0] = False
    # mask[:,-1] = False
    boundary_indices = full_indices[mask].flatten()
    return boundary_indices

def sample_observations(u, obs_count, obs_noise):
    time_size, grid_size = u.shape
    # Get observations at random locations
    rng = np.random.default_rng()
    t_idxs = np.arange(time_size)
    x_idxs = np.arange(grid_size)
    T_idxs, X_idxs = np.meshgrid(t_idxs, x_idxs, indexing='ij')
    all_idxs = np.stack([T_idxs.flatten(), X_idxs.flatten()], axis=1)
    idxs = rng.choice(all_idxs, obs_count, replace=False)
    obs_dict = {tuple(idx): u[tuple(idx)]+obs_noise*np.random.randn() for idx in idxs}
    return obs_dict

def fit_spde_grf(u, obs_dict, X_test, dx, dt, obs_noise, diff_op):
    # Construct precision matrix corresponding to the linear differential operator
    mat = operator_to_matrix(diff_op, u.shape, interior_only=False)
    prior_precision = mat.T @ mat

    return _fit_grf(u, obs_dict, obs_noise, prior_precision)

def _fit_grf(ground_truth, obs_dict, obs_noise, prior_precision):
    # Process args
    shape = ground_truth.shape
    obs_idxs = np.array(list(obs_dict.keys()), dtype=int)
    obs_noise_inv_sq = obs_noise**(-2)

    # Construct posterior precision and posterior shift
    N = np.prod(shape)
    grid_idxs = get_domain_indices(shape)
    boundary_idxs = get_boundary_indices(shape)
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
    posterior_precision_cholesky = cholesky(posterior_precision)
    posterior_mean = posterior_precision_cholesky(posterior_shift).reshape(shape)
    posterior_var = posterior_precision_cholesky.spinv().diagonal().reshape(shape)
    posterior_std = np.sqrt(posterior_var)

    # Compute log marginal likelihood
    P = posterior_precision[obs_mask, :][:, obs_mask]
    L = cholesky(P)
    y = np.array([obs_dict[tuple(idx)] for idx in obs_idxs])
    n = len(y)
    log_marginal_likelihood = 0.5 * (L.logdet() - y.T @ P @ y - n * np.log(2 * np.pi))
    return posterior_mean, posterior_std, log_marginal_likelihood

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
    gp.fit(swap_cols(obs_idx), obs_vals)

    # Make predictions with the GP model over the input grid
    y_pred, std_dev = gp.predict(X_test, return_std=True)
    y_pred = y_pred.reshape(shape)
    std_dev = std_dev.reshape(shape)
    return y_pred, std_dev

def plot_gp_2d(gt, gp_mean, gp_std, obs_idx, output_filename, mean_vmin, mean_vmax,
               std_vmin, std_vmax, diff_vmin, diff_vmax):
    fig, axs = plt.subplots(1, 4, figsize=(15,5.3))
    gtim = axs[0].imshow(gt, vmin=mean_vmin, vmax=mean_vmax)
    axs[0].set_title('ground truth')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('x')
    fig.colorbar(gtim)
    ptim = axs[1].imshow(gp_mean, vmin=mean_vmin, vmax=mean_vmax)
    axs[1].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
    axs[1].set_title('mean')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('x')
    fig.colorbar(ptim)
    ptstdim = axs[2].imshow(gp_std, vmin=std_vmin, vmax=std_vmax)
    axs[2].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
    axs[2].set_title('standard deviation')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('x')
    fig.colorbar(ptstdim)
    diffim = axs[3].imshow(gt - gp_mean, vmin=diff_vmin, vmax=diff_vmax)
    axs[3].set_title('diff')
    axs[3].set_xlabel('time')
    axs[3].set_ylabel('x')
    fig.colorbar(diffim)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)

def create_HTML_animation(x, u, posterior_mean_1, posterior_std_1, posterior_mean_2,
                          posterior_std_2, dt, output_filename):    
    t_steps = u.shape[0]
    # Create animation
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    line_gt1, = ax[0].plot(x, u[0], label='ground truth')
    line_gt2, = ax[1].plot(x, u[0], label='ground truth')
    line_mean, = ax[0].plot(x, posterior_mean_1[0], label='posterior mean')
    line_std = ax[0].fill_between(x, posterior_mean_1[0] - posterior_std_1[0], posterior_mean_1[0] + posterior_std_1[0], facecolor='orange', alpha=0.2, label='posterior std.')
    line_mean_rbf, = ax[1].plot(x, posterior_mean_2[0], c='green', label='posterior mean (rbf)')
    line_std_rbf = ax[1].fill_between(x, posterior_mean_2[0] - posterior_std_2[0], posterior_mean_2[0] + posterior_std_2[0], facecolor='green', alpha=0.2, label='posterior std. (rbf)')
    ax[0].set_ylim([-0.1,1.2])
    ax[1].set_ylim([-0.1,1.2])
    ax[0].set_xlabel("x")
    ax[1].set_xlabel("x")
    ax[0].set_ylabel("u(x, t)")
    ax[1].set_ylabel("u(x, t)")
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')

    def animate(i):
        line_gt1.set_ydata(u[i])
        line_gt2.set_ydata(u[i])
        line_mean.set_ydata(posterior_mean_1[i])
        line_mean_rbf.set_ydata(posterior_mean_2[i])
        ax[0].collections.clear()
        ax[1].collections.clear()
        line_std = ax[0].fill_between(x, posterior_mean_1[i] - posterior_std_1[i], posterior_mean_1[i] + posterior_std_1[i], facecolor='orange', alpha=0.2)
        line_std_rbf = ax[1].fill_between(x, posterior_mean_2[i] - posterior_std_2[i], posterior_mean_2[i] + posterior_std_2[i], facecolor='green', alpha=0.2)
        ax[0].set_title(f'SPDE Kernel t={(dt*i):02.3f}')
        ax[1].set_title(f'RBF Kernel t={(dt*i):02.3f}')
        return line_gt1, line_gt2, line_mean, line_std, line_mean_rbf, line_std_rbf

    anim = animation.FuncAnimation(fig, animate, frames=t_steps//2, interval=300, blit=True)
    anim.save(output_filename, writer='imagemagick', fps=2)
    return HTML(anim.to_jshtml())

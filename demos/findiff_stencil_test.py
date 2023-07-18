from jax.experimental import sparse as jax_sparse
import jax.numpy as jnp
import numpy as np
from scipy import sparse as sp_sparse
from findiff import FinDiff
import matplotlib.pyplot as plt
import warnings

np.set_printoptions(threshold=None)
np.set_printoptions(linewidth=None)

# Test Heat Eqn Op
shape = (8, 8)
alpha = 0.05
dt = 0.1
dx = 0.1

diff_t = FinDiff(0, dt, 1)
diff_x = FinDiff(1, dx, 2)

diff_op = diff_t - alpha * diff_x
diff_mat_findiff = diff_op.matrix(shape).todense()

# Plot all fin. diff. stencils
plt_cols = 8
plt_rows = diff_mat_findiff.shape[0] // plt_cols
plt_size_multiplier = 1.5
fix, ax = plt.subplots(plt_rows, plt_cols, figsize=(plt_size_multiplier * plt_rows, plt_size_multiplier * plt_cols))
ax = ax.flatten()
for i in range(np.prod(shape)):
    ax[i].imshow(diff_mat_findiff[i,:].reshape(shape), vmin=-20, vmax=20)
plt.show()

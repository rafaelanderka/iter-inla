import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load Allen-Cahn eq. data from PINNs examples
data = loadmat("data/PINNs/AC.mat")
uu = data['uu']
xx = data['x'].squeeze()
tt = data['tt'].squeeze()
N_x = xx.shape[0]
N_t = tt.shape[0]
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]

# Plot ground truth data
plt.imshow(uu, origin="lower", extent=[tt.min(), tt.max(), xx.min(), xx.max()])
plt.xlabel("t")
plt.ylabel("x")
plt.tight_layout()
plt.show()

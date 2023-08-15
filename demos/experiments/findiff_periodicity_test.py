import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff

shape = (10, 10)
u = np.zeros(shape)
u[-1,:] = 1

plt.imshow(u, origin="lower")
plt.show()

diff_x = FinDiff(0, 0.1, 1)
u_x = diff_x(u)

plt.imshow(u_x, origin="lower")
plt.show()

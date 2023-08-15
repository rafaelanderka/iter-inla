from time import time
import numpy as np
from sksparse.cholmod import cholesky
from scipy.sparse import identity
from findiff import FinDiff
import matplotlib.pyplot as plt

from spdeinf import util

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

debug = False
diff_op = FinDiff(0, 1, 1) - 0.1 * FinDiff(1, 1, 2)
n_obs = 100
ns = []
precisions = []
shifts = []

print("Computing matrices to factor and systems to solve...")
ks = np.arange(10, 1010 + 1, 100)
for k in ks:
    start_time = time()
    n = k * k
    ns.append(n)
    shape = (k, k)
    mat = util.operator_to_matrix(diff_op, shape, interior_only=False)
    if debug and k == 10:
        M = mat.T @ mat
        print(M.todense())
        non_zero_counts = np.count_nonzero(M.toarray(), axis=1)
        for i, count in enumerate(non_zero_counts):
            print(f"Row {i+1}: {count} non-zero elements")
    precisions.append(mat.T @ mat + identity(n, format='csc'))
    shift = np.zeros(np.prod(shape))
    for i in range(n_obs):
        idx = np.random.randint(n, size=2)
        shift[idx] = 100
    shifts.append(shift)
    print(f'k = {k}, time = {time() - start_time} s')

print("Factoring matrices, solving systems, inverting partially...")
ts = []
for k, precision, shift in zip(ks, precisions, shifts):
    start_time = time()
    posterior_precision_cholesky = cholesky(precision)
    posterior_mean = posterior_precision_cholesky(shift)
    posterior_var = posterior_precision_cholesky.spinv()
    t = time() - start_time
    ts.append(t)
    print(f'k = {k}, time = {t} s')

for n, t in zip(ns, ts):
    print(t/n)

plt.plot(ns, ts)
plt.xlabel("N")
plt.ylabel("t [s]")
plt.title("Time to compute posterior vs. field size")
plt.savefig("figures/complexity_test.png")
plt.show()

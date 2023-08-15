import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def gaussian_pdf(x, mu, var):
    return np.exp(-((x - mu) ** 2) / (2 * var)) / (np.sqrt(2 * np.pi * var))

x = np.linspace(-50, 175, 10000)

var1, mu1 = 25.0, 30.0
y1 = gaussian_pdf(x, mu1, var1)

var2, mu2 = 100.0, 10.0
y2 = gaussian_pdf(x, mu2, var2)

mixture_weights = [0.5, 0.5]

def first_mom_integrd(x, mu, var):
    return x * gaussian_pdf(x, mu, var)

def second_mom_integrd(x, mu, var):
    return (x ** 2) * gaussian_pdf(x, mu, var)

def compute_stats(pdf, a, b):
    first_mom = integrate.quad(lambda x: x * pdf(x), a, b)[0]
    second_mom = integrate.quad(lambda x: (x ** 2) * pdf(x), a, b)[0]
    print(first_mom)
    print(second_mom)

    mean = first_mom
    variance = second_mom - first_mom ** 2

    print("-- Mean:", first_mom)
    print("-- Variance:", second_mom - first_mom ** 2)
    return mean, variance

print("Numerical summary statistics for dist. 1")
compute_stats(lambda x: gaussian_pdf(x, mu1, var1), 0, 60)

print("Numerical summary statistics for dist. 2")
compute_stats(lambda x: gaussian_pdf(x, mu2, var2), -40, 60)

print("Numerical summary statistics for mixture")
compute_stats(lambda x: mixture_weights[0] * gaussian_pdf(x, mu1, var1) + mixture_weights[1] * gaussian_pdf(x, mu2, var2), -40, 60)

print("True summary statistics for mixture")
first_mom_mixture = mixture_weights[0] * mu1  + mixture_weights[1] * mu2
second_mom_mixture = mixture_weights[0] * (var1 + mu1 ** 2) + mixture_weights[1] * (var2 + mu2 ** 2)
var_mixture = second_mom_mixture - first_mom_mixture ** 2
# var_mixture = mixture_weights[0] * var1 + mixture_weights[1] * var2 - mixture_weights[0] * mixture_weights[1] * mu1 * mu2
print("-- Mean:", first_mom_mixture)
print("-- Variance:", var_mixture)

plt.plot(x, y1, ls="--", label="y1")
plt.axvline(mu1, label="mu1")
plt.plot(x, y2, ls="--", label="y2")
plt.axvline(mu2, label="mu2")
plt.legend()
plt.grid()
plt.show()

import numpy as np
from scipy.stats import gaussian_kde, entropy

def kl_divergence(p, q, base=None):
    return entropy(p, q, base=base)

def data_simulator(n, func):
    return func(n)

# Example usage:
n = 10

# Define a function to build 4-dimensional Gaussian simulators
my_gaussian_simulator_builder = lambda m,s: (lambda n: np.random.multivariate_normal(m,s,n))

# Define the means and covariance matrices for the 4-dimensional Gaussians
mean1 = np.zeros(4)
cov1 = np.eye(4)
mean2 = np.zeros(4)
cov2 = 2 * np.eye(4)

# Create the Gaussian simulators
mygsim_0_1 = my_gaussian_simulator_builder(mean1, cov1)
mygsim_0_2 = my_gaussian_simulator_builder(mean2, cov2)

# Generate the data
data1 = data_simulator(n, mygsim_0_1)
data2 = data_simulator(n, mygsim_0_2)

# Estimate the PDFs of the two data simulators
pdf1 = gaussian_kde(data1.T)
pdf2 = gaussian_kde(data2.T)

# Evaluate the PDFs on a grid
x = np.linspace(-5, 5, num=n)
X, Y, Z, W = np.meshgrid(x, x, x, x)
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), W.ravel()])
p = pdf1.evaluate(positions)
q = pdf2.evaluate(positions)

# Calculate the KLD
kld = kl_divergence(p, q)
print(f"The Kullback-Leibler Divergence between the two data simulators is {kld}")


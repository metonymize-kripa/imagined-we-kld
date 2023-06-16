"""
This script calculates the Kullback-Leibler Divergence (KLD) between two 4-dimensional Gaussian data simulators. The KLD is a measure of the difference between two probability distributions.

Here's a brief overview of the script:

1. The `kl_divergence` function computes the KLD between two input probability distributions (p and q) using scipy's entropy function.
2. The `data_simulator` function generates data samples from a function, passed as an argument, that simulates the desired data distribution.
3. The lambda function `my_gaussian_simulator_builder` is used to define 4-dimensional Gaussian simulators, where `m` is the mean and `s` is the covariance matrix.
4. We define the means (`mean1`, `mean2`) and covariance matrices (`cov1`, `cov2`) for the two Gaussian distributions.
5. The Gaussian simulators (`mygsim_0_1`, `mygsim_0_2`) are then created using the `my_gaussian_simulator_builder` function.
6. We generate data samples (`data1`, `data2`) from the two Gaussian simulators.
7. The Probability Density Functions (PDFs) of the two sets of data are estimated using scipy's `gaussian_kde` function.
8. The PDFs are then evaluated on a 4-dimensional grid.
9. Finally, the script calculates and prints the KLD between the two simulated Gaussian distributions.

Note: The parameters (mean and covariance) of the Gaussian distributions and the number of samples (n) are set for this specific run 
but can be modified as needed. 

WARNING: Using more than a small value of n (around 10) could cause the program to freeze due to the amount of memory needed.
"""


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


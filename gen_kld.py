"""
This script primarily demonstrates how to estimate the Kullback-Leibler Divergence (KLD) between two data distributions. 

In this example, two Gaussian data distributions with different standard deviations (but the same mean) are simulated. 
The Kullback-Leibler Divergence between the two distributions is then computed.

1. First, the necessary libraries, numpy and scipy.stats, are imported.
2. Next, the function `kl_divergence(p, q, base=None)` is defined, which calculates the Kullback-Leibler divergence between two probability distributions `p` and `q`. 
3. The function `data_simulator(n, func)` is defined, which takes in an integer `n` and a function `func` and returns an array of `n` simulated data points based on `func`.
4. An example usage is then provided, where the number of data points to be simulated `n` is set to 1000. Two Gaussian data simulators are defined with the same mean (0) and different standard deviations (1 and 2, respectively).
5. These data simulators are then used to generate `n` data points.
6. The probability density functions (PDFs) of these two sets of simulated data are estimated using the Gaussian Kernel Density Estimation (KDE) method.
7. A linear space `x` is created which spans the range of the data from both simulators.
8. The estimated PDFs are evaluated over this linear space to generate probability distributions `p` and `q`.
9. The Kullback-Leibler Divergence between the two probability distributions `p` and `q` is then calculated using the `kl_divergence` function defined earlier.
10. Finally, the calculated Kullback-Leibler Divergence is printed to the console.

In summary, this script showcases how to simulate data, estimate PDFs, and compute the Kullback-Leibler Divergence between two distributions.
"""

import numpy as np
from scipy.stats import gaussian_kde, entropy

def kl_divergence(p, q, base=None):
    """Calculate Kullback-Leibler divergence between two distributions.
    
    Args:
        p, q (array-like): Input arrays.
        base (float, optional): The logarithmic base to use when computing the entropy. Defaults to `e` (natural logarithm).
        
    Returns:
        float: The Kullback-Leibler divergence of `q` from `p`.
    """
    return entropy(p, q, base=base)

def data_simulator(n, func):
    """Simulate data using a given function.
    
    Args:
        n (int): Number of data points to simulate.
        func (callable): Function to use for data simulation.
        
    Returns:
        array: Simulated data.
    """
    return func(n)

# Example usage:
n = 1000
my_gaussian_simulator_builder = lambda m,s: (lambda n: np.random.normal(m,s,n))

mygsim_0_1 = my_gaussian_simulator_builder(0,1)
mygsim_0_2 = my_gaussian_simulator_builder(0,2)

data1 = data_simulator(n, mygsim_0_1)
data2 = data_simulator(n, mygsim_0_2)

# Estimate the PDFs of the two data simulators
pdf1 = gaussian_kde(data1)
pdf2 = gaussian_kde(data2)

# Evaluate the PDFs on a linear space
x = np.linspace(min(min(data1), min(data2)), max(max(data1), max(data2)), num=n)
p = pdf1.evaluate(x)
q = pdf2.evaluate(x)

# Calculate the KLD
kld = kl_divergence(p, q)
print(f"The Kullback-Leibler Divergence between the two data simulators is {kld}")


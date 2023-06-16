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


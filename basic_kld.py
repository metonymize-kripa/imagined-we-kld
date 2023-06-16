"""
This Python script is used to compute the Kullback-Leibler (KL) Divergence between two Gaussian (Normal) distributions. 
KL divergence measures how one probability distribution is different from a second, reference probability distribution. 
In this script, each Gaussian distribution is defined by a mean ('mu') and a standard deviation ('sigma').

The function 'kl_divergence' calculates the KL divergence using the formula for two Gaussian distributions: 
KL(P||Q) = log(σ2/σ1) + (σ1² + (μ1 - μ2)²) / (2σ2²) - 1/2. 
Here, P and Q represent the two distributions, with μ1, σ1 and μ2, σ2 as their respective mean and standard deviation values.

An example usage of this function is shown where the mean and standard deviation of two Gaussian distributions are defined. 
The KL divergence between these two distributions is calculated and printed. 
The output value indicates the amount of information loss when distribution Q is used to approximate distribution P. 
A lower value indicates a better approximation."""


import numpy as np

def kl_divergence(mu1, sigma1, mu2, sigma2):
    """Compute the Kullback-Leibler Divergence between two Gaussian distributions.
    
    Args:
        mu1 (float): Mean of the first Gaussian distribution.
        sigma1 (float): Standard deviation of the first Gaussian distribution.
        mu2 (float): Mean of the second Gaussian distribution.
        sigma2 (float): Standard deviation of the second Gaussian distribution.
        
    Returns:
        float: The Kullback-Leibler Divergence between the two Gaussian distributions.
    """
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

# Example usage:
mu1 = 0
sigma1 = 1
mu2 = 0
sigma2 = 2

kld = kl_divergence(mu1, sigma1, mu2, sigma2)
print(f"The Kullback-Leibler Divergence between the two Gaussian distributions is {kld}")


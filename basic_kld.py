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


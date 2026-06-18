import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    x= np.where(x==0, 1-p, p)
    return(x, p, p*(1-p))
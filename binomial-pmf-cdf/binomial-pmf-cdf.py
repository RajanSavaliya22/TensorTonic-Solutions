import numpy as np
import scipy
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    p = np.asarray(p, dtype=float)
    pmf = scipy.special.comb(n,k)*(p**k)*((1-p)**(n-k))
    cmf = np.sum(
        [
            scipy.special.comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(k+1)
        ]
    )

    return (pmf, cmf)
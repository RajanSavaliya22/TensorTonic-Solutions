import numpy as np
import math
def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    pmf = (np.exp(-lam))*(lam**k)/math.factorial(k)
    cdf = np.sum(
        [
            (np.exp(-lam))*(lam**i)/math.factorial(i) for i in range(k+1)
        ]
    )
    return (pmf,cdf)
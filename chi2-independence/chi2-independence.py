import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    C = np.asarray(C, dtype=float)
    E = np.outer(np.sum(C, axis=1),np.sum(C,axis=0))/ (np.sum(C))
    chi2 = np.sum((C - E) ** 2 / E)
    return (chi2, E)

    
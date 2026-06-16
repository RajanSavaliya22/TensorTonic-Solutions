import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    try:
        X = np.asarray(X, dtype=float)
    except:
        return None
    N = X.shape[0]

    if X.ndim !=2:
        return None
        
    if N <2:
        return None
    
    
    mean =np.mean(X, axis=0, keepdims=True)
    X_cen = X - mean
    cov = X_cen.T@X_cen / (N-1)

    std = np.sqrt(np.diag(cov))

    den = np.outer(std,std)

    corr = cov/den
    corr[den == 0] = np.nan

    return corr

    
    
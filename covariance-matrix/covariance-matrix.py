import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    try:
        X = np.asarray(X, dtype=float)
    except:
        return None

    if X.ndim !=2:
         return None
    N = X.shape[0]
    if N < 2:
        return None

    mean = np.mean(X, axis=0, keepdims=True)
    X_cen = X - mean

    cov_matrix = (
        (X_cen.T@X_cen) / (N-1)
    )

    return cov_matrix
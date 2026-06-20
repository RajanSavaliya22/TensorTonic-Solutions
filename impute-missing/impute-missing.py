import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.array(X, dtype=float, copy=True)

    if strategy not in ('mean', 'median'):
        raise ValueError("strategy must be 'mean' or 'median'")
    if X.ndim == 1:
        mask = np.isnan(X)

        if np.all(mask):
            fill_value = 0.0
        else:
            observed = X[~mask]
            fill_value = np.mean(observed) if strategy == 'mean' else np.median(observed)

        X[mask] = fill_value
    else:
        for j in range(X.shape[1]):
            col = X[:, j]
            mask = np.isnan(col)
    
            if np.all(mask):
                fill_value = 0.0
            else:
                observed = col[~mask]
                if strategy == 'mean':
                    fill_value = np.mean(observed)
                else:
                    fill_value = np.median(observed)
    
            col[mask] = fill_value

    return X
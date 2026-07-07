import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """

    if rng is None:
        rng = np.random.default_rng()

        
    indices = np.arange(N)
    
    if shuffle:
        rng.shuffle(indices)

        
    fold_sizes = np.full(k, N//k, dtype=int)
    fold_sizes[:N%k] += 1

    folds = []
    current=0

    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size

        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        folds.append((train_idx, val_idx))
        current= stop

    return folds
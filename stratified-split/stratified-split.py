import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype= float)

    if rng is None:
        rng = np.random.default_rng()

    train_indices= []
    test_indices = []

    for cls in np.unique(y):

        indices = np.where(y==cls)[0]
        indices = indices.copy()
        rng.shuffle(indices)

        n = len(indices)
        test_split = int(round(n*test_size))

        if n>1:
            test_split = min(test_split, n-1)

        test_indices.extend(indices[:test_split])
        train_indices.extend(indices[test_split:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    return (X[train_indices], X[test_indices], y[train_indices], y[test_indices])
     

    
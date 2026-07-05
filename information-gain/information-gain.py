import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y, dtype=int)
    h_y = _entropy(y)
    N= len(y)
    value, count = np.unique(y, return_counts=True)
    l_value = [y[i] for i in range(len(y)) if split_mask[i] == value[0]]
    r_value = [y[i] for i in range(len(y)) if split_mask[i] == value[1]]
    n_l = len(l_value)
    n_r = len(r_value)
    if n_l == 0 or n_r == 0:
        return 0.0
    ig = h_y - ((n_l*_entropy(l_value) + n_r*_entropy(r_value))/N)
    return ig

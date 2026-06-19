import numpy as np
from collections import Counter
from scipy import stats

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)
    median = np.median(x)
    flat = x.flatten()
    value, count = np.unique(x, return_counts=True)
    idx = np.argmax(count)
    return (mean, median, value[idx])
    
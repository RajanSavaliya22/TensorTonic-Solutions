import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x , keepdims=True)
    var = np.sum((x-mean)**2)/(n-1)

    return (var, np.sqrt(var))
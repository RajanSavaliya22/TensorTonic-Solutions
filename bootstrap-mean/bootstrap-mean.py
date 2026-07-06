import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    x = np.asarray(x, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    means = []
    for i in range(n_bootstrap):
        samples = x[rng.integers(0, len(x), size=len(x))]
        mean = np.mean(samples)
        means.append(mean)

    b_mean = np.mean(means)
    alpha = 1 - ci
    lower= np.quantile(means, alpha/2)
    upper = np.quantile(means, 1 - (alpha/2))
    means = np.asarray(means, dtype=float)
    return (means, lower, upper)
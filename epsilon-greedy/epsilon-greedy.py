import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    if rng is None:
        rng = np.random.default_rng()
    a = rng.random()
    if a < epsilon:
        idx = rng.integers(0, len(q_values))
        return idx
    else:
        return np.argmax(q_values)

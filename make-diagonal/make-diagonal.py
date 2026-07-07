import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    v = np.asarray(v, dtype=float)
    arr = np.zeros((len(v),len(v)))
    for i in range(len(v)):
        arr[i][i] = v[i]
   
    return arr
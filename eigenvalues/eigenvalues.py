import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix= np.asarray(matrix)
    except:
        return None
        
    if matrix.ndim == 1:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.shape[0] == 0:
        return np.array([])
    eigen_val = np.linalg.eigvals(matrix)

    idx =  np.lexsort((eigen_val.imag, eigen_val.real))
    return eigen_val[idx]
    
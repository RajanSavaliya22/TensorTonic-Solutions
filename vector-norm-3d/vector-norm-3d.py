import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.asarray(v, dtype=float)
    original_dim=v.ndim
    if v.ndim==1:
        v= v[None,:]

    norm = np.sqrt(np.sum(v**2, axis=1))

    if original_dim==1:
        return norm[0]
    return norm
        
    
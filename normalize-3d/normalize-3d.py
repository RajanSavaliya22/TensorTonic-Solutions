import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v= np.asarray(v, dtype=float)

    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return np.zeros_like(v) if norm == 0 else v / norm

    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm!=0)
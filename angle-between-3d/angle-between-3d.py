import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)

    num = v@w
    den = norm_v*norm_w
    if norm_v ==0 or norm_w==0:
        return np.nan
    cos_theta = num / den  
    cos_theta = max(-1, min(cos_theta, 1))
    theta = np.arccos(cos_theta)
    return theta
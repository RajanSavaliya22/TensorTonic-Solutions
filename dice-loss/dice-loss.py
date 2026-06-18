import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    d_coeff = 2*(np.sum(p*y) + eps) / (np.sum(p) + np.sum(y) + eps)

    return np.maximum(0, 1- d_coeff)
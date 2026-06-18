import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)

    if anchor.ndim == 1:
        anchor = anchor[None,:]
   
    D_p = np.sum((anchor-positive)**2, axis=1)
    D_n = np.sum((anchor-negative)**2, axis=1)

    losses =  np.maximum(0, D_p - D_n + margin)

    return np.mean(losses)
    
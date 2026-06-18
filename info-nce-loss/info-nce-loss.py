import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    S = np.dot(Z1, Z2.T) / temperature
    S_scaled = S - np.max(S, axis=1, keepdims=True)
    loss = -np.mean(
        np.log(
            np.exp(np.diag(S_scaled)) 
            /
            np.sum(np.exp(S_scaled), axis=1)
        )
    ) 

    return loss
    
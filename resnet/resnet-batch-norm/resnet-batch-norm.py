import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    x = np.asarray(x, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)
    eps = 1e-7

    if mode == "post":
        h = x @ W1

        mean1 = np.mean(h, axis=0, keepdims=True)
        var1 = np.var(h, axis=0, keepdims=True)
        h = (h - mean1) / np.sqrt(var1 + eps)
        h = gamma1 * h + beta1
        h = np.maximum(0,h)

        h = h @ W2
        mean2 = np.mean(h, axis=0, keepdims=True)
        var2 = np.var(h, axis=0, keepdims=True)
        h = (h - mean2) / np.sqrt(var2 + eps)
        h = gamma2 * h + beta2
        output = np.maximum(0,h + x)
        
    elif mode == "pre":
        h = x
        mean1 = np.mean(h, axis=0, keepdims=True)
        var1 = np.var(h, axis=0, keepdims=True)
        h = (h - mean1) / np.sqrt(var1 + eps)
        h = gamma1 * h + beta1
        h = np.maximum(0,h)
        h = h@W1

        mean2 = np.mean(h, axis=0, keepdims=True)
        var2 = np.var(h, axis=0, keepdims=True)
        h = (h - mean2) / np.sqrt(var2 + eps)
        h = gamma2 * h + beta2
        h = np.maximum(0,h)
        h = h@W2
        output = h+x
        
    return {'output':output, 'mode':mode}
        
        
        



import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HERE
    x = np.asarray(x, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    W3 = np.asarray(W3, dtype=float)
    Ws = np.asarray(Ws, dtype=float)

    h = relu(x@W1) 
    z = relu(h@W2)
    y = z@W3
    return relu(y + x@Ws)

def relu(x):
    return np.maximum(0,x)

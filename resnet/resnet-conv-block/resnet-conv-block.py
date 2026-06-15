import numpy as np

def conv_block(x, W1, W2, Ws):
    x = np.asarray(x, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    Ws = np.asarray(Ws, dtype=float)

    h = np.maximum(0, x @ W1)

    return np.maximum(0, h @ W2 + x @ Ws)

import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g, dtype=float)

    norm = np.linalg.norm(g)

    if norm<=0 or max_norm<=0:
        return g
    else:
        if norm <= max_norm:
            return g
        else:
            output = g*(max_norm/norm)
            return output
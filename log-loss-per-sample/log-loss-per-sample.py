import math
import numpy as np
def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    y_pred = np.clip(y_pred, 1e-15, 1- 1e-15)

    loss = -(y_true*np.log(y_pred) + (1-y_true)*np.log((1-y_pred)))

    return loss.tolist()
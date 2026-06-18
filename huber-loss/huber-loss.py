import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    e = y_true - y_pred
    loss = np.where(np.abs(e)<=delta, 0.5*(e**2), delta*(np.abs(e) - 0.5*delta))
    return np.mean(loss)
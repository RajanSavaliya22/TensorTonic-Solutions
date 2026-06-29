import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cf = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cf[t, p] += 1

    if normalize == 'none':
        return cf
    elif normalize == 'true':
        r_sum = np.sum(cf, axis=1, keepdims=True)
        return cf/r_sum
    elif normalize == 'pred':
        c_sum = np.sum(cf, axis=0, keepdims=True)
        return cf/c_sum
    elif normalize == 'all':
        return cf/np.sum(cf)
    
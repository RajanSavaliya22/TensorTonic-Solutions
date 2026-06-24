import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    order = np.lexsort((1-y_true, -y_score))
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1-y_true_sorted)

    P = tp[-1]  
    N = fp[-1]

    distinct = np.where(np.diff(y_score_sorted))[0]
    idx = np.r_[distinct, len(y_score_sorted) - 1]

    tpr = tp[idx] / P if P > 0 else np.zeros(len(idx))
    fpr = fp[idx] / N if N > 0 else np.zeros(len(idx))
    thresholds = y_score_sorted[idx]

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    if len(tpr) == 1 or tpr[-1] != 1 or fpr[-1] != 1:
        tpr = np.r_[tpr, 1]
        fpr = np.r_[fpr, 1]
        thresholds = np.r_[thresholds, -np.inf]

    return fpr, tpr, thresholds
    
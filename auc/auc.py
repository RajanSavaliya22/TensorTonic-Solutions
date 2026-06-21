import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    if len(tpr) != len(fpr):
        return None
    if len(tpr) < 2 or len(tpr) < 2:
        return None
    return np.trapezoid(tpr,fpr)
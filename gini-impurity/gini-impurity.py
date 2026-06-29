import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    y_left = np.asarray(y_left,dtype=float)
    y_right = np.asarray(y_right,dtype=float)

    values_l, counts_l = np.unique(y_left, return_counts=True)
    values_r, counts_r = np.unique(y_right, return_counts=True)
    
    if len(values_l) <= 1 and len(values_r) <= 1:
        return 0.0
    p_l = counts_l/len(y_left)
    gini_l = 1 - np.sum(p_l**2)
    p_r = counts_r/len(y_right)
    gini_r = 1 - np.sum(p_r**2)

    total = len(y_left) + len(y_right)

    return (len(y_left)*gini_l + len(y_right)*gini_r)/total
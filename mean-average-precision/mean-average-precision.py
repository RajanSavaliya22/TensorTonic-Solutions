import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        
        order = np.argsort(-y_score)

        rel = y_true[order]

        R = rel.sum()
        
        if k is not None:
            rel = rel[:k]


        if R == 0:
            ap_per_query.append(0.0)
            continue

        tp_cumsum = np.cumsum(rel)
        ranks = np.arange(1,len(rel) + 1)
        precision = tp_cumsum / ranks

        ap = np.sum(precision*rel)/R
        ap_per_query.append(ap)

    ap_per_query = np.asarray(ap_per_query, dtype=float)
    map_value = ap_per_query.mean() if len(ap_per_query) else 0.0

    return map_value, ap_per_query
    
import numpy as np
def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here

    if system_type == "classification":
        y_pred = np.asarray(y_pred, dtype=int)
        y_true = np.asarray(y_true, dtype=int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp+fp)
        recall = tp / (tp + fn)
        f1 = 2*precision*recall / (precision+recall)
        return [ ('accuracy',accuracy),
                 ('f1', f1), 
                 ('precision',precision), 
                 ('recall',recall)]

    elif system_type == 'regression':
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        mae = np.sum(abs(y_true - y_pred)) / len(y_pred)
        rmse = np.sqrt(np.sum((y_true-y_pred)**2)/len(y_pred))
        return [('mae',mae), ('rmse', rmse)]

    elif system_type == 'ranking':
        k = 3
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        # Indices of predictions sorted by score (highest first)
        top_k_idx = np.argsort(y_pred)[::-1][:k]
    
        relevant_in_top_k = np.sum(y_true[top_k_idx] == 1)
        total_relevant = np.sum(y_true == 1)
    
        precision_at_3 = relevant_in_top_k / k if k > 0 else 0.0
        recall_at_3 = (
            relevant_in_top_k / total_relevant
            if total_relevant > 0
            else 0.0
        )
    
        return sorted([
            ("precision_at_3", precision_at_3),
            ("recall_at_3", recall_at_3),
        ])
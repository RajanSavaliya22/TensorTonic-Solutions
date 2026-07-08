import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q = np.asarray(Q, dtype=float)
    Q_new = Q.copy()
    
    Q_new[s][a] += alpha*(r + gamma*max(Q[s_next]) - Q[s][a])
    return Q_new
import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    A = np.zeros(len(states))
    G = 0.0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + (gamma)*G
        A[i] = G - V[states[i]]
    return A
        

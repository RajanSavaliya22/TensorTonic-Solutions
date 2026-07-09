import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    V = np.zeros(n_states)
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)

    for episode in episodes:
        G = 0
        returns = [0]*len(episode)

        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = reward + gamma*G
            returns[t] = G

        visited = set()

        for t, (state, reward) in enumerate(episode):
            if state not in visited:
                visited.add(state)
                returns_sum[state] += returns[t]
                returns_count[state] += 1

        for s in range(n_states):
            if returns_count[s] > 0:
                V[s] = returns_sum[s]/returns_count[s]

    return V
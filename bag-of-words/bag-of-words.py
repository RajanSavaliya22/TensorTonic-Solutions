import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    vocab_index = {w: i for i, w in enumerate(vocab)}
    bow = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        idx = vocab_index.get(token)
        if idx is not None:
            bow[idx] += 1

    return bow
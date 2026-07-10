def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    bigram_counts = {}
    first_counts = {} 

    for i in range(len(tokens)-1):
        bigram = (tokens[i], tokens[i+1])
        w1 = tokens[i]
        first_counts[w1] = first_counts.get(w1, 0) + 1
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    vocab = sorted(set(tokens))
    V = len(vocab)
    prob_dict = {}

    for w1 in vocab:
        denom = first_counts.get(w1, 0) + V
        for w2 in vocab:
            count = bigram_counts.get((w1,w2),0)

            prob_dict[(w1,w2)] = (count + 1)/ denom
    return (bigram_counts, prob_dict)
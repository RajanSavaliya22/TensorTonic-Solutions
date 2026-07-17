def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    n = len(actual_tokens)

    cross_entropy = 0

    for probs, token in zip(prob_distributions, actual_tokens):
        p = probs[token]
        cross_entropy -= math.log(p)

    cross_entropy /= n

    return math.exp(cross_entropy)
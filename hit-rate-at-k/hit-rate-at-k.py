def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    number_of_hits = 0.0

    for rec, gt in zip(recommendations, ground_truth):
        if gt[0] in rec[:k]:
            number_of_hits+=1.0

    return number_of_hits/ len(recommendations)
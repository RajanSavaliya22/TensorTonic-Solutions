import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    dcg = 0
    idcg = 0
    sorted_rs = sorted(relevance_scores, reverse = True)
    for i in range(min(k, len(relevance_scores))):
        dcg+= (((2**relevance_scores[i]) - 1)/math.log2(i+2)) 
        idcg+= (((2**sorted_rs[i]) - 1)/math.log2(i+2))

    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg
        
    
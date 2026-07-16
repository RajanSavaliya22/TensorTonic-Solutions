import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """

    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)
    
    if len(rater1) == 0 or len(rater2) ==0 :
        return 
        
    label1, freq1 = np.unique(rater1, return_counts=True)
    label2, freq2 = np.unique(rater2, return_counts=True)

    p = [0]*len(label1)
    for i in range(len(label1)):
        if label1[i] == label2[i]:
            p[i] = freq1[i]*freq2[i] / (len(rater1)**2) 
            
    pc = sum(p)
    p0 = sum([1 for i in range(len(rater1)) if rater1[i]==rater2[i]]) / len(rater1)

    k = (p0 - pc) / (1-pc) if pc != 1 else p0

    return k

    
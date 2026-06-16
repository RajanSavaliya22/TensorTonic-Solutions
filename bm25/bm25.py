import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    df_t = {}
    
    N = len(docs)
    
    doc_lengths = []
    
    for doc in docs:
        doc_lengths.append(len(doc))
        for token in set(doc):
            df_t[token] = df_t.get(token , 0 ) + 1

    average_doclen = np.mean(doc_lengths)
    
    idf_t = {}
    
    for token in df_t.keys():
        df = df_t[token]
        idf_t[token] =math.log(
            ((N - df + 0.5) / (df + 0.5)) + 1
        )

    scores = []
    
    for doc in docs:
        tf = {}
        
        D = len(doc)
        
        score = 0.0
            
        for token in doc:
            tf[token] = tf.get(token, 0) + 1
        
        for q_token in query_tokens:
            freq = tf.get(q_token,0)

            if freq == 0:
                continue
                
            score += (
                (idf_t.get(q_token, 0)*freq*(k1+1)) / 
                (freq + k1*(1-b + b*(D/average_doclen)))
            )
        scores.append(score)
    return np.array(scores)
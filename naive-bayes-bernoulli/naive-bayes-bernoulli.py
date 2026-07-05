import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    # Write code here
    X_train = np.asarray(X_train, dtype=int)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=int)

    value, count = np.unique(y_train, return_counts=True)
    log_prior = np.log(count / len(X_train))
    
    p = np.zeros((len(value), X_train.shape[1]))
    for i in range(len(value)):
        for j in range(len(X_train)):
            if y_train[j] == value[i]:
                p[i] +=  X_train[j] 
        
        one = np.ones(p[i].shape)
        p[i] = (p[i] +1) / (count[i] + 2)
        
    log_likelihood = (
        log_prior + 
        X_test@np.log(p).T +
        (1-X_test)@np.log(1-p).T
    )    


    return log_likelihood
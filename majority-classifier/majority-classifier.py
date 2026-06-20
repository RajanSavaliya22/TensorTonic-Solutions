import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    value, count = np.unique(y_train, return_counts=True)

    idx = np.argmax(count)
    return [value[idx]]*len(X_test)
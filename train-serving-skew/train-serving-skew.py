import numpy as np
def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    output = {}

    for key in train_dist:
        train = np.array(train_dist[key], dtype=np.float64)+eps
        serving = np.array(serving_dist[key], dtype=np.float64)+eps
        term = (serving - train) * np.log(serving / train)
        psi = float(np.sum(term))

        output[key] = {
            "psi": psi,
            "skewed": psi >= threshold,
        }

    return output
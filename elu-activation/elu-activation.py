import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    x = np.asarray(x)
    output = [0]*len(x)
    for i in range(len(x)):
        if x[i] > 0:
            output[i] = x[i]
        else:
            output[i] = alpha*(np.exp(x[i]) - 1)
    return output
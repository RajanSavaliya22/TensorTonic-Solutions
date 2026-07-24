import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    x = np.asarray(x,dtype=float)
    selu = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            selu[i] = lam*x[i]
        else:
            selu[i] = lam*alpha*(np.exp(x[i]) - 1)
    return selu

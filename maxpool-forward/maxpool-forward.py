import numpy as np
def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.asarray(X)

    H_out = (X.shape[0] - pool_size) // stride + 1
    W_out = (X.shape[1] - pool_size) // stride + 1

    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            window = X[
                    i*stride : i*stride + pool_size,
                    j*stride: j*stride + pool_size
                ]
            output[i,j] = np.max(window)
    return output.tolist()
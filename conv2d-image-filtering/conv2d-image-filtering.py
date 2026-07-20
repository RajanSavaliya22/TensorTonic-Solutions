import numpy as np
def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    image = np.asarray(image, dtype=int)
    kernel = np.asarray(kernel,dtype=int)
    H, W = image.shape
    kh, kw = kernel.shape
    pad_image = np.pad(image, 
                   pad_width=((padding, padding), (padding, padding)),
                   mode='constant', 
                   constant_values=0)
    H_out = ((H + 2*padding - kh)//stride) + 1
    W_out = ((W + 2*padding - kw)//stride) + 1

    output = np.zeros((H_out,W_out))

    for i in range(H_out):
        for j in range(W_out):
            r = i*stride
            c = j*stride
            output[i,j] = np.sum(pad_image[r:r+kw, c:c+kh]*kernel)
    return output.tolist()
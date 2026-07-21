import numpy as np
def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    center = size // 2

    output = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            output[i,j] = np.exp(-(x**2 + y**2)/(2*(sigma**2)))
    output = output / np.sum(output.flatten())
    return output.tolist()
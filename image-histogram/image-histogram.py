import numpy as np
def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    hist = [0]*256
    image = np.asarray(image, dtype=int)
    for pixel in image.flat:
        hist[pixel]+=1
    return hist
    
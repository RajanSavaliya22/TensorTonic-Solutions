import numpy as np
def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    image = np.asarray(image,dtype=float)
    result = []
    for i in range(image.shape[0]):
        x= []
        for j in range(image.shape[1]):
            x.append(
                
                    0.299*image[i][j][0]+
                    0.587*image[i][j][1]+
                    0.114*image[i][j][2])
        result.append(x)
    return result
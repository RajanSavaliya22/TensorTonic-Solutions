import numpy as np
def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    image = np.asarray(image,dtype=float)
    pad_image = np.pad(
        image,
        pad_width = ((1,1),(1,1)),
        mode = 'constant',
        constant_values= 0
    )

    Kx = np.asarray(
        [
            [-1.0,0.0,1.0],
            [-2.0,0.0,2.0],
            [-1.0,0.0,1.0]
        ]
    )
    Ky = np.asarray(
        [
            [-1.0,-2.0,-1.0],
            [0.0,0.0,0.0],
            [1.0,2.0,1.0]
        ]
    )
    H, W = image.shape
    output = np.zeros_like(image)

    for i in range(H):
        for j in range(W):
            Gx = np.sum(pad_image[i:i+3, j:j+3]*Kx)
            Gy = np.sum(pad_image[i:i+3, j:j+3]*Ky)
            output[i,j] = np.sqrt((Gx**2) + (Gy**2))

    return output.tolist()
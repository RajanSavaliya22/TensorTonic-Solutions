import numpy as np
def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    image = np.asarray(image)

    H, W = image.shape
    cx = (W - 1) / 2
    cy = (H - 1) / 2

    theta = np.deg2rad(angle_degrees)

    output = np.zeros_like(image)

    for i in range(H):
        for j in range(W):

            dy = i - cy
            dx = j - cx

            src_x = cx + dx * np.cos(theta) - dy * np.sin(theta)
            src_y = cy + dx * np.sin(theta) + dy * np.cos(theta)

            isrc = round(src_y)
            jsrc = round(src_x)

            if 0 <= isrc < H and 0 <= jsrc < W:
                output[i, j] = image[isrc, jsrc]

    return output.tolist()
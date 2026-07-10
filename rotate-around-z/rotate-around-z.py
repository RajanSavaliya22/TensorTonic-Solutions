import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    points = np.asarray(points) 
    # if points.ndim == 1:
    #     output = np.zeros(len(points))
    #     output[0] = points[0]*np.cos(theta) - points[1]*np.sin(theta)
    #     output[1] = points[0]*np.sin(theta) + points[1]*np.cos(theta)
    #     output[2] = points[2]
    #     return output
    # else:
        # output = np.zeros(points.shape)
        # for i in range(points.shape[0]):
        #     output[i,0] = points[i,0]*np.cos(theta) - points[i,1]*np.sin(theta)
        #     output[i,1] = points[i,0]*np.sin(theta) + points[i,1]*np.cos(theta)
        #     output[i,2] = points[i,2]
        # return output
    original_shape = points.shape
    original_dim = points.ndim
    if points.ndim == 1:
        points = points[None, :]
    R = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0,0, 1]
        ]
    )

    output = R@points.T
    if original_dim == 1:
        return output.reshape(original_shape)
    return output.T
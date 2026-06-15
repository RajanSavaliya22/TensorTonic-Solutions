import numpy as np

def compute_gradient_with_skip(gradients_F, x):
    gradients_F = np.asarray(gradients_F, dtype=float)
    x = np.asarray(x, dtype=float)

    result = x

    for grad in gradients_F:
        result = (np.eye(grad.shape[0]) + grad).T @ result

    return result


def compute_gradient_without_skip(gradients_F, x):
    gradients_F = np.asarray(gradients_F, dtype=float)
    x = np.asarray(x, dtype=float)

    result = x

    for grad in gradients_F:
        result = grad.T @ result

    return result
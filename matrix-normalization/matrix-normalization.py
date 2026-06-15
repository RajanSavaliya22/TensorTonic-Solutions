import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        matrix = np.asarray(matrix, dtype=float)
    except:
        return None
    if matrix.ndim not in (1, 2):
        return None

    if norm_type not in ('l1', 'l2', 'max'):
        return None

    if axis is not None:
        if axis >= matrix.ndim or axis < -matrix.ndim:
            return None

    if norm_type == 'l1':
        norm = np.sum(np.abs(matrix),
                      axis=axis,
                      keepdims=(axis is not None))

    elif norm_type == 'l2':
        norm = np.sqrt(
            np.sum(matrix ** 2,
                   axis=axis,
                   keepdims=(axis is not None))
        )

    else:  # max
        norm = np.max(np.abs(matrix),
                      axis=axis,
                      keepdims=(axis is not None))

    # avoid division by zero
    norm = np.where(norm == 0, 1, norm)

    return matrix / norm
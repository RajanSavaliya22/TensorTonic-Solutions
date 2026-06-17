import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    x = np.asarray(x, dtype=float)

    # Initial layer
    x = np.maximum(0, x @ conv1)

    # Block 1 (identity shortcut)
    h = np.maximum(0, x @ W1_b1)
    h = h @ W2_b1
    x = np.maximum(0, h + x)

    # Block 2 (projection shortcut)
    h = np.maximum(0, x @ W1_b2)
    h = h @ W2_b2

    shortcut = x @ Ws_b2

    x = np.maximum(0, h + shortcut)

    # Classifier
    logits = x @ fc

    return logits

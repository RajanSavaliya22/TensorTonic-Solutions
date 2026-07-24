import math
def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    loss= 0
    for i in range(len(predictions)):
        if i == target:
            q = (1 - epsilon) + epsilon/len(predictions)
        else:
            q = epsilon/len(predictions)

        loss += q*math.log(predictions[i])
    return -loss
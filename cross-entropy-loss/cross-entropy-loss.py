import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    inds = len(y_true)
    p_pred = y_pred[np.arange(inds), y_true]
    loss = -np.mean(np.log(p_pred))
    return loss
import numpy as np

def log_loss(y_true: list, y_pred: list) -> float:
    """
    Returns the mean binary log loss.
    """
    rows = len(y_true)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)

    cros_entp = -1/rows * np.sum(y_true* np.log(y_pred) + (1-y_true)* np.log(1 - y_pred))

    return round(float(cros_entp),4)
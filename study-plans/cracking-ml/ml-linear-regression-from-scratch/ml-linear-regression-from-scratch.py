import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    rows = len(X)
    cols = len(X[0])
    
    w = np.zeros(cols)
    b = 0

    X = np.array(X)
    y = np.array(y)
    
    for epoch in range(epochs):
        # same as np.dot(X, w)
        y_pred = X @ w + b
        error = y_pred - y # (rows,) and X:(rows,cols)
    
        dw = (2/rows) * (X.T @ error) # becarefull that transpose doesnt work on list and they need to become numpy array first
        db = 2/rows * np.sum(error)
        
        w = w - lr * dw
        b = b - lr * db
    
    return (w, b)

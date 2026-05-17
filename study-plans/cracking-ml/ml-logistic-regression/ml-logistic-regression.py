import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    rows = len(X)
    cols = len(X[0])

    w = np.zeros(cols)# w:(cols), X:(rows,cols) 
    b = 0

    X = np.array(X)
    y = np.array(y)

    for epoch in range(n_iters):
        z = (X @ w + b)
        y_pred = 1/ (1 + np.exp(-z))
    
        error = y_pred - y # error: (rows,)
    
        dw = 1/rows * X.T @ error
        db = 1/rows * np.sum(error)
    
        w = w - lr * dw
        b = b - lr * db

    return (w,b)
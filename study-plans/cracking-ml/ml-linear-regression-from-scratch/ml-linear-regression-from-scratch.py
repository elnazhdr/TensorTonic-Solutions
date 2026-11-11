import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    rows = len(X)
    cols = len(X[0])

    w = np.zeros(cols)
    b = 0

    X = np.array(X)
    y = np.array(y)

    for epoch in range(epochs):
        y_hat = X@w + b
        error = y_hat - y

        # if we consider MSE for loss formula
        # Loss = sigma(y_hat-y)^2
        dw = 2/rows * X.T @ (error) # d Loss/dw
        db = 2/rows * np.sum(error) # d Loss/db
        
        w = w - lr * dw
        b = b - lr * db

    return (w, b)
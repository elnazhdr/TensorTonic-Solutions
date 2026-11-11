import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    rows = len(X)
    cols = len(X[0])

    w = np.zeros(cols)
    b = 0

    X = np.array(X)
    y = np.array(y)

    for epoch in range(n_iters):
        y_hat = X @ w + b
        y_prob = 1/(1+np.exp(-y_hat))
        error = y_prob - y

        dw = 1/rows * X.T @ (error)
        db = 1/rows * np.sum(error)
        
        w = w - lr* dw
        b = b - lr* db
        
    return (w, b)
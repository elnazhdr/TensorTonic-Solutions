import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    predictions = []

    for x in X_test:
        d = np.sqrt(np.sum((X_train - x) ** 2, axis = 1))
        
        nn_idx = np.argsort(d)[:k]
        
        nn_labels = y_train[nn_idx]

        pred = np.argmax(np.bincount(nn_labels))

        predictions.append(pred)

    return predictions    
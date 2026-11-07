import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    if m == 0:
        return

    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros
    Xt = np.transpose(X)

    for i in range(iterations):
        preds = np.matmul(X, theta)
        err = np.subtract(preds, y)
        Xte = np.matmul(Xt, err)
        grad = np.multiply(Xte, 1/m)
        ag = np.multiply(grad, alpha)
        theta = np.subtract(theta, ag)

    return theta.flatten()

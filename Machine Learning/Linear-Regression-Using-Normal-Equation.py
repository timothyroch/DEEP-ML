import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	arrX = np.array(X)
	arrY = np.array(y)
	xt = np.transpose(arrX)
	xtx = np.matmul(xt, arrX)
	xtx_inv = np.linalg.inv(xtx)
	tetha = np.matmul(np.matmul(xtx_inv, xt), arrY)
	return tetha.tolist()

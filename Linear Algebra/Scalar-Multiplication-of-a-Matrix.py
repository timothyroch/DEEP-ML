import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:

	if len(matrix) == 0 or len(matrix[0]) == 0:
		return []
	
	arr = np.array(matrix)

	return np.multiply(arr, scalar).tolist()

import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:

	if len(matrix) == 0 or len(matrix[0]) == 0:
		return 0

	arr = np.array(matrix)

	md = 1
	if mode == "column":
		md = 0
	
	return np.mean(arr, md).tolist()

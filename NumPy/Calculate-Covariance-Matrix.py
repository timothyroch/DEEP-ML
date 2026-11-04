import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	
	if len(vectors)==0 or len(vectors[0])==0:
		return []

	arr = np.array(vectors)


	return np.cov(arr, rowvar=True).tolist()

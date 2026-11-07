import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	
	mean = np.mean(data, 0, keepdims=True)
	stdD = np.std(data, 0, keepdims=True)
	ones = np.ones(data.shape)
	standardized_data = np.round(np.multiply(np.subtract(data, mean), 1/stdD), 4)

	mini = np.min(data, 0)
	maxi = np.max(data, 0)
	diff = maxi-mini
	normalized_data = np.round(np.multiply(np.subtract(data, mini), 1/diff), 4)

	return standardized_data, normalized_data

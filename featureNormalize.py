import numpy as np
def featureNormalize(X):
	m = len(X)
	mu = np.mean(X)
	sigma = np.std(X)
	X_norm = (X-mu)/sigma
	return X_norm , mu , sigma


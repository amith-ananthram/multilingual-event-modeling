import numpy as np
from sklearn.cluster import KMeans

_2PI_LOG = np.log(2*np.pi)

class CovarianceMatrix:
	def __init__(self, matrix):
		# if we happen to generate a singular covariance 
		# matrix, we increase its diagonal entries by enough 
		# to raise its lowest eigenvalue to 1
		if np.linalg.det(matrix) == 0:
			eigs, _ = np.linalg.eigh(matrix)
			matrix += (1 - eigs[0]) * np.eye(matrix.shape[0])
		self.matrix = matrix
		self.det = np.linalg.det(matrix)
		self.inv = np.linalg.inv(matrix)


def get_kmeans_assignments(embeddings, num_means):
	k_means = KMeans(num_means)
	k_means = k_means.fit(embeddings)
	return k_means.predict(embeddings), k_means.cluster_centers_


def logsumexp(x):
	c = x.max()
	return c + np.log(np.sum(np.exp(x - c)))


def calculate_multivariate_normal_logpdf(x, mean, cov):
		logpdf = (-x.shape[0]/2) * _2PI_LOG
		logpdf += (-1/2) * np.log(cov.det)

		x_dev = x - mean
		logpdf += (-1/2) * (x_dev.transpose() @ cov.inv @ x_dev)
		return logpdf
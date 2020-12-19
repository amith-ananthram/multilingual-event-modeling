import numpy as np
from sklearn.cluster import KMeans
from scipy.special import gammaln

_2PI_LOG = np.log(2*np.pi)

class CovarianceMatrix:
	def __init__(self, matrix, det=None, inv=None):
		# if we happen to generate a singular covariance 
		# matrix, we increase its diagonal entries by enough 
		# to raise its lowest eigenvalue to 1
		if np.linalg.det(matrix) < 1:
			eigs, _ = np.linalg.eigh(matrix)
			matrix += (1 - eigs[0]) * np.eye(matrix.shape[0])
		self.matrix = matrix
		self.det = np.linalg.det(matrix) if det is None else det
		self.inv = np.linalg.inv(matrix) if inv is None else inv

	def scaled(cov_matrix, scaling_factor):
		matrix = scaling_factor * cov_matrix.matrix
		det = (scaling_factor ** cov_matrix.matrix.shape[0]) * cov_matrix.det
		inv = (1 / scaling_factor) * cov_matrix.inv
		return CovarianceMatrix(matrix, det, inv)


def get_kmeans_assignments(embeddings, num_means):
	k_means = KMeans(num_means)
	k_means = k_means.fit(embeddings)
	return k_means.predict(embeddings), k_means.cluster_centers_


def logsumexp(x):
	max_x = x.max()
	return max_x + np.log(np.sum(np.exp(x - max_x)))


def calculate_multivariate_normal_logpdf(x, mean, cov):
		logpdf = (-x.shape[0]/2) * _2PI_LOG
		logpdf += (-1/2) * np.log(cov.det)

		x_dev = x - mean
		logpdf += (-1/2) * (x_dev.transpose() @ cov.inv @ x_dev)
		return logpdf


def calculate_multivariate_t_logpdf(x, mean, shape, df):
	dim = mean.size

	vals, vecs = np.linalg.eigh(shape)
	logdet     = np.log(vals).sum()
	valsinv    = np.array([1./v for v in vals])
	U          = vecs * np.sqrt(valsinv)
	dev        = x - mean
	maha       = np.square(np.dot(dev, U)).sum(axis=-1)

	t = 0.5 * (df + dim)
	A = gammaln(t)
	B = gammaln(0.5 * df)
	C = dim/2. * np.log(df * np.pi)
	D = 0.5 * logdet
	E = -t * np.log(1 + (1./df) * maha)

	logpdf = A - B - C - D + E
	
	return logpdf
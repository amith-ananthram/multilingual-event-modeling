import unittest
import numpy as np
from scipy import stats

import utils

class UtilsTest(unittest.TestCase):


	def test_logsumexp(self):
		print("WRITE LOGSUMEXP TEST")


	def test_calculate_multivariate_normal_logpdf(self):
		x = np.random.rand(100)
		mean = np.random.rand(100)
		rand_mat = np.random.rand(100, 100)
		cov = utils.CovarianceMatrix(rand_mat.transpose() @ rand_mat)

		self.assertAlmostEqual(
			stats.multivariate_normal.logpdf(x, mean, cov.matrix),
			utils.calculate_multivariate_normal_logpdf(
				x, mean, cov
			),
			places=3
		)
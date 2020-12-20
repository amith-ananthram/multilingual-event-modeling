import sys
import mock
import unittest
import numpy as np
from scipy import stats

import utils
from preprocess import TrainingData
from samplers.naive_sampler import NaiveSampler

class NaiveSamplerTest(unittest.TestCase):


	def get_default_sampler(self, training_data, num_nav_topics=3, num_ne_topics=3):
		return NaiveSampler([], 1, training_data, num_nav_topics=num_nav_topics, 
			nav_topic_mean_prior_means=[
				np.mean(training_data.nav_embeddings, axis=0) for _ in range(num_nav_topics)],
			nav_topic_mean_prior_kappa=1,
			nav_topic_covariance_prior_dof=training_data.nav_embeddings.shape[1] + 2,
			nav_topic_covariance_prior_scale=3 * np.eye(training_data.nav_embeddings.shape[1], dtype=np.float64),
			nav_article_topic_proportions_prior_alpha=np.ones(num_nav_topics),
			nav_initialization='r',
			num_ne_topics=num_ne_topics,
			ne_topic_vocab_prior_alpha=np.ones(len(training_data.ne_vocab)),
			ne_article_topic_proportions_prior_alpha=np.ones(num_ne_topics)
		)


	def get_default_sampler_with_dummy_data(self, num_nav_topics):
		training_data = TrainingData(
			articles=["DUMMY_STRING" for _ in range(5)],
			nav_vocab=["nav1", "nav2", "nav3", "nav4"],
			nav_embeddings=np.stack([
				np.full(10, fill_value=1),
				np.full(10, fill_value=2),
				np.full(10, fill_value=3),
				np.full(10, fill_value=4),
				np.full(10, fill_value=5)
			]),
			article_navs=[
				[0, 0, 1],
				[0, 1, 2],
				[0, 1, 2, 3],
				[0, 4],
				[0, 1, 1]
			],
			ne_vocab=["ne1", "ne2", "ne3"],
			article_nes=[[], [], [], [], []]
		)
		sampler = self.get_default_sampler(training_data, num_nav_topics=num_nav_topics)

		return sampler


	def test_calculate_nav_intermediate_values(self):
		sampler = self.get_default_sampler_with_dummy_data(4)
		sampler.nav_article_nav_assignments = [
			[], 									# prior sample, unused
			[										# current sample
				[0, 0, 1],
				[0, 1, 0],
				[0, 0, 1, 1],
				[0, 1],
				[1, 0, 2]
			]										
		]
		sampler.calculate_nav_intermediate_values()

		# topic 0 stats
		topic0_stats = sampler.nav_topic_stats[0]
		np.testing.assert_array_equal(
			np.stack([
				np.full(10, fill_value=1), np.full(10, fill_value=1), 
				np.full(10, fill_value=1), np.full(10, fill_value=3),
				np.full(10, fill_value=1), np.full(10, fill_value=2),
				np.full(10, fill_value=1), np.full(10, fill_value=2)
			]),
			np.stack(topic0_stats[0])
		)
		self.assertEqual(8, topic0_stats[1])
		np.testing.assert_array_equal(
			np.full(10, fill_value=(1+1+1+3+1+2+1+2)), topic0_stats[2]
		)

		# topic 1 stats
		topic1_stats = sampler.nav_topic_stats[1]
		np.testing.assert_array_equal(
			np.stack([
				np.full(10, fill_value=2), np.full(10, fill_value=2), 
				np.full(10, fill_value=3), np.full(10, fill_value=4),
				np.full(10, fill_value=5), np.full(10, fill_value=1)
			]),
			np.stack(topic1_stats[0])
		)
		self.assertEqual(6, topic1_stats[1])
		np.testing.assert_array_equal(
			np.full(10, fill_value=(2+2+3+4+5+1)), topic1_stats[2]
		)

		# topic 2 stats
		topic2_stats = sampler.nav_topic_stats[2]
		np.testing.assert_array_equal(
			np.stack([
				np.full(10, fill_value=2)
			]),
			np.stack(topic2_stats[0])
		)
		self.assertEqual(1, topic2_stats[1])
		np.testing.assert_array_equal(
			np.full(10, fill_value=2), topic2_stats[2]
		)

		# topic 3 stats (empty topic)
		topic3_stats = sampler.nav_topic_stats[3]
		self.assertEqual(0, len(topic3_stats[0]))
		self.assertEqual(0, topic3_stats[1])
		np.testing.assert_array_equal(
			np.zeros(10), topic3_stats[2]
		)

		# article topic stats
		np.testing.assert_array_equal(
			np.array(
				[
					[2, 1, 0, 0],
					[2, 1, 0, 0],
					[2, 2, 0, 0],
					[1, 1, 0, 0],
					[1, 1, 1, 0]
				]
			),
			sampler.nav_article_topic_counts
		)


	@mock.patch('samplers.naive_sampler.dirichlet')
	def test_sample_nav_article_proportions(self, mock_dirichlet):
		sampler = self.get_default_sampler_with_dummy_data(4)
		sampler.nav_article_topic_counts = [None, np.array([2, 1, 0, 0]), None, None]
	
		expected_article_proportions = stats.dirichlet.rvs(
			np.ones(4) + np.array([2, 1, 0, 0])
		)
		mock_dirichlet.rvs.return_value = expected_article_proportions

		actual_article_proportions = sampler.sample_nav_article_proportions(1)

		self.assertEqual(1, len(mock_dirichlet.rvs.call_args[0]))
		np.testing.assert_array_equal(
			np.ones(4) + np.array([2, 1, 0, 0]),
			mock_dirichlet.rvs.call_args[0][0]
		)

		np.testing.assert_array_equal(expected_article_proportions[0], actual_article_proportions)


	@mock.patch('samplers.naive_sampler.choice')
	def test_sample_nav_article_nav_assignments(self, mock_choice):
		sampler = self.get_default_sampler_with_dummy_data(4)
		sampler.nav_article_proportions = [
			[],									# prior sample, unused
			[
				None,
				None,	
				None,
				np.array([1, 2, 3, 4]),			# navs = 0, 4
				None
			]
		]
		sampler.nav_topic_covariances = [
			[],									# prior sample, unused
			[
				utils.CovarianceMatrix(2*np.eye(10)),
				utils.CovarianceMatrix(3*np.eye(10)),
				utils.CovarianceMatrix(4*np.eye(10)),
				utils.CovarianceMatrix(5*np.eye(10))
			]
		]
		sampler.nav_topic_means = [
			[],									# prior sample, unused
			[
				np.full(10, fill_value=1.1),
				np.full(10, fill_value=2.1),
				np.full(10, fill_value=3.1),
				np.full(10, fill_value=4.1)
			]
		]

		expected_proportions = np.array([
			[
				1 * stats.multivariate_normal.pdf(
					np.full(10, fill_value=1), sampler.nav_topic_means[-1][0], sampler.nav_topic_covariances[-1][0].matrix),
				2 * stats.multivariate_normal.pdf(
					np.full(10, fill_value=1), sampler.nav_topic_means[-1][1], sampler.nav_topic_covariances[-1][1].matrix),
				3 * stats.multivariate_normal.pdf(
					np.full(10, fill_value=1), sampler.nav_topic_means[-1][2], sampler.nav_topic_covariances[-1][2].matrix),
				4 * stats.multivariate_normal.pdf(
					np.full(10, fill_value=1), sampler.nav_topic_means[-1][3], sampler.nav_topic_covariances[-1][3].matrix)
			],
			[
				1 * stats.multivariate_normal.pdf(
					np.full(10, fill_value=5), sampler.nav_topic_means[-1][0], sampler.nav_topic_covariances[-1][0].matrix),
				2 * stats.multivariate_normal.pdf(
					np.full(10, fill_value=5), sampler.nav_topic_means[-1][1], sampler.nav_topic_covariances[-1][1].matrix),
				3 * stats.multivariate_normal.pdf(
					np.full(10, fill_value=5), sampler.nav_topic_means[-1][2], sampler.nav_topic_covariances[-1][2].matrix),
				4 * stats.multivariate_normal.pdf(
					np.full(10, fill_value=5), sampler.nav_topic_means[-1][3], sampler.nav_topic_covariances[-1][3].matrix)
			]
		])
		expected_probability1 = expected_proportions[0] / expected_proportions[0].sum()
		expected_probability2 = expected_proportions[1] / expected_proportions[1].sum()

		mock_choice.side_effect = [1, 3]
		updated_assignments = sampler.sample_nav_article_nav_assignments(3)

		self.assertEqual([1, 3], updated_assignments)

		self.assertEqual(2, len(mock_choice.call_args_list))
		np.testing.assert_array_almost_equal(expected_probability1, mock_choice.call_args_list[0][1]['p'])
		np.testing.assert_array_almost_equal(expected_probability2, mock_choice.call_args_list[1][1]['p'])




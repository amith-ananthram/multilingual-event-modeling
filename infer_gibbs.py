import numpy as np
from scipy.stats import invwishart
from numpy.random import dirichlet, multivariate_normal
from multiprocessing.pool import ThreadPool

import preprocess

NUM_WORKERS = 8

# clean up representation
	# for each article, we want a list of words ids (include repeats)
# make sure it's everywhere
# cache get_unnormalized_topic_word_probability
# run it on a toy example
# implement sampling for rest
# implement log joint

# TOMORROW:
	# - filter corpus to subset
	# - experiment with Gibbs
	# - experiment with AEVB
	# - write up results, come up posterior predictives
	# - work on cholesky

class TrainingData:
	def __init__(self, articles, noun_and_verb_vocabulary, noun_and_verb_embeddings, article_noun_and_verb_counts):
		self.articles = articles 
		self.noun_and_verb_vocabulary = noun_and_verb_vocabulary
		self.noun_and_verb_embeddings = noun_and_verb_embeddings
		self.article_noun_and_verb_counts = article_noun_and_verb_counts
		self.article_noun_and_verb_vocabularies = [
			[nav_id for nav_id, count in enumerate(article_noun_and_verb_counts[article_id]) if count > 0]
			for article_id in range(article_noun_and_verb_counts.shape[0])
		]

	def get_nav_embedding(article_id, article_nav_id):
		return self.noun_and_verb_embeddings[
			self.article_noun_and_verb_vocabularies[article_id][article_nav_id]
		]

SAMPLER_MODES = [
	'naive',
	'cholesky-collapsed',
	'cholesky-collapsed-aliased'
]

class CovarianceMatrix:
	def __init__(self, matrix):
		self.matrix = matrix
		self.matrix_det = np.linalg.det(matrix)
		self.matrix_inv = np.linalg.inv(matrix)


class Sampler:
	def __init__(self, mode, training_data, 
		num_nav_topics, nav_topic_mean_prior_mean, nav_topic_mean_prior_covariance,
		nav_topic_covariance_prior_dof, nav_topic_covariance_prior_scale,
		nav_article_topic_proportions_prior_alpha
	):
		self.mode = mode
		self.training_data = training_data
		self.num_nav_topics = num_nav_topics
		# multivariate-Gaussian prior on our topic means
		self.nav_topic_mean_prior_mean = nav_topic_mean_prior_mean
		self.nav_topic_mean_prior_covariance = CovarianceMatrix(nav_topic_mean_prior_covariance)
		self.nav_topic_mean_prior_covariance_inv_mean_product = \
			np.matmul(self.nav_topic_mean_prior_covariance.inv, self.topic_mean_prior_mean)
		# inverse-Wishart prior on our topic covariances
		self.nav_topic_covariance_prior_dof = nav_topic_covariance_prior_dof
		self.nav_topic_covariance_prior_scale = nav_topic_covariance_prior_scale
		# Dirichlet prior on our article topic proportions
		self.nav_article_topic_proportions_prior_alpha = nav_article_topic_proportions_prior_alpha

	def sample_nav_topic_mean_and_covariance(topic_id):
		topic_nav_count = len(self.nav_topic_assignments[topic_id])
		topic_nav_sum = sum([
			self.training_data.get_nav_embedding(article_id, article_nav_id) \
			for (article_id, article_nav_id) in self.nav_topic_assignments[topic_id]
		])

		##
		# re-sample the topic mean
		##

		topic_mean_covariance_inv = \
			self.nav_topic_mean_prior_covariance.inv + topic_nav_count * topic_covariance.inv
		topic_mean_covariance = np.linalg.inv(topic_mean_covariance_inv)
		topic_mean_mean = np.matmul(
			topic_mean_covariance,
			self.nav_topic_mean_prior_covariance_inv_mean_product + \
			np.matmul(self.nav_topic_covariances[-1][topic_id].inv, topic_nav_sum)
		)
		# since we already know the inverse, can we pass that?
		topic_mean = multivariate_normal(
			topic_mean_mean,
			topic_mean_covariance
		)

		##
		# re-sample the topic covariance
		##

		topic_covariance_dof = topic_nav_count + self.nav_topic_covariance_prior_dof
		topic_covariance_scale = self.nav_topic_covariance_prior_scale + \
			sum([np.matmul(topic_nav - topic_mean, (topic_nav - topic_mean).transpose()) for topic_nav in topic_navs])
		topic_covariance = invwishart.rvs(topic_covariance_dof, topic_covariance_scale)

		return topic_mean, CovarianceMatrix(topic_covariance)

	def sample_nav_article_proportions(article_id):
		return dirichlet(
			self.nav_article_topic_proportions_prior_alpha \
			+ self.nav_article_topic_counts[article_id]
		)

	# cache
	def get_unnormalized_topic_word_probability(topic_id, word_id):
		return self.nav_topic_covariances[-1][topic_id].det ** (-1/2) \
			* np.exp(np.matmul())

	def sample_nav_article_word_assignments(article_id):
		nav_article_proportions = self.nav_article_proportions[-1][article_id]
		unnormalized = [
			[nav_article_proportions[topic_id] * get_unnormalized_topic_word_probability(topic_id, word_id) for topic in range(self.num_nav_topics)]
			for 
		]


	def run(self, num_iterations):
		##
		# SAMPLED VALUES
		##

		# num iterations x num topics x word embedding dim
		self.nav_topic_means = []
		# num iterations x num topics x word embedding dim^2
		self.nav_topic_covariances = []
		# num iterations x num articles x num topics
		self.nav_article_proportions = []
		# num iterations x num articles x num navs per document
		self.nav_article_word_assignments = []

		##
		# INTERMEDIATE VALUES
		##

		# num topics x [(article_id, article_word_id)]
		self.nav_topic_assignments = defaultdict(list)
		# num articles x num topics 
		self.nav_article_topic_counts = []

		##
		# MONITORING VALUES
		##

		self.log_joints = [calculate_log_joint()]

		for epoch in range(num_iterations):
			# sample topic means & covariances
			with ThreadPool(NUM_WORKERS) as p:
				updated_nav_topic_means_and_covariances = p.map(
					sample_nav_topic_mean_and_covariance, range(self.num_nav_topics)
				)
				self.nav_topic_means.append(
					[nav_updated_topic_mean for (nav_updated_topic_mean, _) in updated_nav_topic_means_and_covariances])
				self.nav_topic_covariances.append(
					[nav_updated_topic_covariance for (_, nav_updated_topic_covariance) in updated_nav_topic_means_and_covariances])

				self.nav_article_proportions.append(p.map(
					sample_nav_article_proportions, 
					range(len(self.training_data.article_noun_and_verb_counts))
				))

				self.nav_article_word_assignments.append(p.map(
					sample_nav_article_word_assignments,
					range(len(self.training_data.article_noun_and_verb_counts))
				))

				self.nav_topic_assignments = defaultdict(list)
				self.nav_article_topic_counts = np.zeros(len(self.training_data.article_noun_and_verb_counts), self.num_nav_topics)

				# update nav_topic_assignments
				# update nav_article_topic_counts

			# check convergence (calculate joint?)

			# sample word assignments (prop to proportions_ik * |Sigma_k|^(-1/2)e^{(-1/2)(w - mu_k)^T\Sigma^{-1}(x - \mu)})

if __name__ == '__main__':
	# articles, noun_and_verb_vocabulary, noun_and_verb_embeddings, article_noun_and_verb_counts = \
	# 	preprocess.preprocess_articles(langs, datetime(1996, 9, 1), datetime(1996, 9, 2))

	sampler = Sampler("naive", None, 7, None, None)
	sampler.run(1)
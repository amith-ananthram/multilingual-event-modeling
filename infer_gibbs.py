import numpy as np
from datetime import datetime
from numpy.random import choice
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from scipy.stats import dirichlet, invwishart, multivariate_normal

import preprocess
from preprocess import Article

NUM_WORKERS = 8

# run it on a toy example
# implement sampling for rest (allow running w/ only one set or the other set)
# move samplers into their own directory

# TOMORROW:
	# - filter corpus to subset (use most common named entities + random)
	# - experiment with Gibbs
	# - experiment with AEVB
	# - write up results, come up posterior predictive checks
	# - work on cholesky

class TrainingData:
	def __init__(self, articles, nav_vocab, nav_embeddings, article_navs, ne_vocab, article_nes):
		self.articles = articles 
		self.nav_vocab = nav_vocab
		self.nav_embeddings = nav_embeddings
		self.article_navs = article_navs
		self.ne_vocab = ne_vocab
		self.article_nes = article_nes

	def get_nav_embedding(self, article_id, article_nav_id):
		return self.nav_embeddings[self.article_navs[article_id][article_nav_id]]


SAMPLER_MODES = [
	'naive',
	'cholesky-collapsed',
	'cholesky-collapsed-aliased'
]


class CovarianceMatrix:
	def __init__(self, matrix):
		# if we happent to generate a singular covariance 
		# matrix, we increase its diagonal entries by enough 
		# to raise its lowest eigenvalue to 1
		if np.linalg.det(matrix) < 1:
			eigs, _ = np.linalg.eigh(matrix)
			matrix += (1 - eigs[0]) * np.eye(matrix.shape[0])
		self.matrix = matrix
		self.det = np.linalg.det(matrix)
		self.inv = np.linalg.inv(matrix)


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
			self.nav_topic_mean_prior_covariance.inv @ self.nav_topic_mean_prior_mean
		# inverse-Wishart prior on our topic covariances
		self.nav_topic_covariance_prior_dof = nav_topic_covariance_prior_dof
		self.nav_topic_covariance_prior_scale = nav_topic_covariance_prior_scale
		# Dirichlet prior on our article topic proportions
		self.nav_article_topic_proportions_prior_alpha = nav_article_topic_proportions_prior_alpha


	def sample_nav_topic_mean_and_covariance(self, topic_id):
		topic_navs, topic_nav_count, topic_nav_sum = self.nav_topic_stats[topic_id]

		##
		# re-sample the topic mean
		##

		topic_mean_covariance_inv = \
			self.nav_topic_mean_prior_covariance.inv + topic_nav_count * self.nav_topic_covariances[-1][topic_id].inv

		topic_mean_covariance = np.linalg.inv(topic_mean_covariance_inv)

		topic_mean_mean = topic_mean_covariance @ (
			self.nav_topic_mean_prior_covariance_inv_mean_product + 
			(self.nav_topic_covariances[-1][topic_id].inv @ topic_nav_sum)
		)

		topic_mean = multivariate_normal.rvs(
			topic_mean_mean,
			topic_mean_covariance
		)

		##
		# re-sample the topic covariance
		##

		topic_covariance_dof = topic_nav_count + self.nav_topic_covariance_prior_dof

		# the einsum below produces the sum of the outer products
		# of the deviations of each word vector from the topic mean
		if len(topic_navs) > 0:
			topic_nav_devs = np.stack(topic_navs) - topic_mean 
			topic_covariance_scale = self.nav_topic_covariance_prior_scale + \
				np.einsum('ki,kj->ij', topic_nav_devs, topic_nav_devs)
		else:
			topic_covariance_scale = self.nav_topic_covariance_prior_scale

		topic_covariance = invwishart.rvs(
			topic_covariance_dof, np.linalg.inv(topic_covariance_scale))

		return topic_mean, CovarianceMatrix(topic_covariance)


	def sample_nav_article_proportions(self, article_id):
		return dirichlet.rvs(
			self.nav_article_topic_proportions_prior_alpha \
			+ self.nav_article_topic_counts[article_id]
		)[0]


	def logsumexp(self, x):
		c = x.max()
		return c + np.log(np.sum(np.exp(x - c)))


	def calculate_multivariate_normal_logpdf(self, x, mean, cov):
		logpdf = (-x.shape[0]/2) * np.log(np.pi)
		logpdf += (-1/2) * np.log(cov.det)

		x_dev = x - mean
		logpdf += (-1/2) * (x_dev @ cov.inv @ x_dev.transpose())
		return logpdf


	def calculate_topic_nav_logprob(self, topic_id, nav_id):
		cache_key = ("topic_nav_logprob", topic_id, nav_id)
		if cache_key not in self.cache:
			# nav = self.training_data.nav_embeddings[nav_id]
			# cov = self.nav_topic_covariances[-1][topic_id]
			# mean_dev = nav - self.nav_topic_means[-1][topic_id]

			# self.cache[cache_key] = cov.det ** (-1/2) \
				# * np.exp(-(1/2) * mean_dev @ cov.inv @ mean_dev.transpose())
			self.cache[cache_key] = self.calculate_multivariate_normal_logpdf(
				self.training_data.nav_embeddings[nav_id], 
				self.nav_topic_means[-1][topic_id], 
				self.nav_topic_covariances[-1][topic_id]
			)

		return self.cache[cache_key]


	def sample_nav_article_nav_assignments(self, article_id):
		nav_article_log_proportions = np.log(
			self.nav_article_proportions[-1][article_id])

		updated_assignments = []
		for nav_id in self.training_data.article_navs[article_id]:
			assignment_proportions = nav_article_log_proportions + np.array([
				self.calculate_topic_nav_logprob(topic_id, nav_id) \
				for topic_id in range(self.num_nav_topics)
			])
			assignment_probabilities = np.exp(
				assignment_proportions - self.logsumexp(assignment_proportions))
			updated_assignments.append(choice(self.num_nav_topics, p=assignment_probabilities))

		return updated_assignments

	# def get_unnormalized_topic_nav_probability(self, topic_id, nav_id):
	# 	cache_key = ("unnorm_topic_nav_prob", topic_id, nav_id)
	# 	if cache_key not in self.cache:
	# 		nav = self.training_data.nav_embeddings[nav_id]
	# 		cov = self.nav_topic_covariances[-1][topic_id]
	# 		mean_dev = nav - self.nav_topic_means[-1][topic_id]

	# 		self.cache[cache_key] = cov.det ** (-1/2) \
	# 			* np.exp(-(1/2) * mean_dev @ cov.inv @ mean_dev.transpose())

	# 	return self.cache[cache_key]


	# # log sum exp?
	# def sample_nav_article_nav_assignments(self, article_id):
	# 	nav_article_proportions = self.nav_article_proportions[-1][article_id]
	# 	unnormalized = np.array([
	# 		[nav_article_proportions[topic_id] * self.get_unnormalized_topic_nav_probability(
	# 			topic_id, nav_id) for topic_id in range(self.num_nav_topics)] \
	# 		for nav_id in self.training_data.article_navs[article_id]
	# 	])
	# 	normalized = unnormalized / unnormalized.sum(axis = 1, keepdims=True)
	# 	return [choice(self.num_nav_topics, p=p) for p in normalized]


	def calculate_intermediate_values(self):
		# num topics x [(# of assignments, sum of embeddings)]
		self.nav_topic_stats = defaultdict(lambda: [[], 0, np.zeros(self.training_data.nav_embeddings.shape[1])])
		# num articles x num topics 
		self.nav_article_topic_counts = np.zeros((len(self.training_data.articles), self.num_nav_topics))

		for article_id in range(len(self.training_data.article_navs)):
			for article_nav_id in range(len(self.training_data.article_navs[article_id])):
				nav_topic_assignment = self.nav_article_nav_assignments[-1][article_id][article_nav_id]

				nav = self.training_data.nav_embeddings[
					self.training_data.article_navs[article_id][article_nav_id]]
				self.nav_topic_stats[nav_topic_assignment][0].append(nav)
				self.nav_topic_stats[nav_topic_assignment][1] += 1
				self.nav_topic_stats[nav_topic_assignment][2] += nav
				self.nav_article_topic_counts[article_id][nav_topic_assignment] += 1

		for topic_id in range(self.num_nav_topics):
			if self.nav_topic_stats[topic_id][1] == 0:
				print("empty topic: %s" % topic_id)
				# print(self.cache["test"])
				# raise Exception("WAH")


	def calculate_log_joint(self):
		log_joint = 0
		for nav_topic_id in range(self.num_nav_topics):
			log_joint += self.calculate_multivariate_normal_logpdf(
				self.nav_topic_means[-1][nav_topic_id], 
				self.nav_topic_mean_prior_mean, 
				self.nav_topic_mean_prior_covariance
			)
			log_joint += invwishart.logpdf(
				self.nav_topic_covariances[-1][nav_topic_id].matrix,
				self.nav_topic_covariance_prior_dof,
				self.nav_topic_covariance_prior_scale
			)

		for article_id in range(len(self.training_data.articles)):
			log_joint += dirichlet.logpdf(
				self.nav_article_proportions[-1][article_id],
				self.nav_article_topic_proportions_prior_alpha
			)

			for article_nav_id, nav_id in enumerate(self.training_data.article_navs[article_id]):
				nav_topic_assignment = self.nav_article_nav_assignments[-1][article_id][article_nav_id]
				log_joint += np.log(self.nav_article_proportions[-1][article_id][nav_topic_assignment])
				log_joint += self.calculate_topic_nav_logprob(nav_topic_assignment, nav_id)

				# can we calculate this in a batch for every word assigned to a topic?
				# log_joint += self.calculate_multivariate_normal_logpdf(
				# 	self.training_data.nav_embeddings[nav_id],
				# 	self.nav_topic_means[-1][nav_topic_assignment],
				# 	self.nav_topic_covariances[-1][nav_topic_assignment]
				# )

		return log_joint


	def run(self, num_iterations):
		##
		# SAMPLED VALUES
		##

		# num iterations x num topics x word embedding dim
		self.nav_topic_means = []
		# num iterations x num topics x word embedding dim^2
		self.nav_topic_covariances = [[
			self.nav_topic_mean_prior_covariance for _ in range(self.num_nav_topics)
		]]
		# num iterations x num articles x num topics
		self.nav_article_proportions = []
		# num iterations x num articles x num navs per document
		self.nav_article_nav_assignments = [[
			[choice(self.num_nav_topics) for _ in range(len(self.training_data.article_navs[article_id]))]
			for article_id in range(len(self.training_data.articles))
		]]

		##
		# MONITORING VALUES
		##

		self.log_joints = []

		print("%s: beginning sampling..." % datetime.now())

		for epoch in range(num_iterations):
			# general purpose cache for 
			# memoizing expensive ops
			self.cache = {}
			self.calculate_intermediate_values()

			# with ThreadPool(NUM_WORKERS) as p:
			# 	# re-sample nav topic means & covariances
			# 	updated_nav_topic_means_and_covariances = p.map(
			# 		self.sample_nav_topic_mean_and_covariance, range(self.num_nav_topics)
			# 	)
			# 	self.nav_topic_means.append(
			# 		[nav_updated_topic_mean for (nav_updated_topic_mean, _) in updated_nav_topic_means_and_covariances])
			# 	self.nav_topic_covariances.append(
			# 		[nav_updated_topic_covariance for (_, nav_updated_topic_covariance) in updated_nav_topic_means_and_covariances])

			# 	# re-sample nav article proportions
			# 	self.nav_article_proportions.append(p.map(
			# 		self.sample_nav_article_proportions, 
			# 		range(len(self.training_data.articles))
			# 	))

			# 	# re-sample nav article nav assignments
			# 	self.nav_article_nav_assignments.append(p.map(
			# 		self.sample_nav_article_nav_assignments,
			# 		range(len(self.training_data.articles))
			# 	))

			print("%s: Resampling topic means" % (datetime.now()))

			updated_nav_topic_means = []
			updated_nav_topic_covariances = []
			for nav_topic_id in range(self.num_nav_topics):
				updated_nav_topic_mean, updated_nav_topic_covariance = self.sample_nav_topic_mean_and_covariance(nav_topic_id)

				updated_nav_topic_means.append(updated_nav_topic_mean)
				updated_nav_topic_covariances.append(updated_nav_topic_covariance)
			self.nav_topic_means.append(updated_nav_topic_means)
			self.nav_topic_covariances.append(updated_nav_topic_covariances)

			print("%s: Resampling article proportions" % (datetime.now()))

			updated_nav_article_proportions = []
			for article_id in range(len(self.training_data.articles)):
				updated_nav_article_proportions.append(self.sample_nav_article_proportions(article_id))
			self.nav_article_proportions.append(updated_nav_article_proportions)

			print("%s: Resampling topic assignments" % (datetime.now()))

			updated_nav_article_nav_assignments = []
			for article_id in range(len(self.training_data.articles)):
				updated_nav_article_nav_assignments.append(self.sample_nav_article_nav_assignments(article_id))
			self.nav_article_nav_assignments.append(updated_nav_article_nav_assignments)

			self.log_joints.append(self.calculate_log_joint())

			if epoch % 10 == 0:
				for topic_id in range(self.num_nav_topics):
					top_nav_ids = np.argsort(-np.array([
						self.calculate_topic_nav_logprob(topic_id, nav_id) \
						for nav_id in range(len(self.training_data.nav_vocab))
					]))[:5]
					top_article_ids = np.argsort(
						-np.stack(updated_nav_article_proportions)[:, topic_id])[:5]
					print("topic %s: %s; %s" % (
							topic_id, 
							",".join([
								self.training_data.nav_vocab[nav_id][1] for nav_id in top_nav_ids]), 
							",".join(map(str, top_article_ids))
						)
					)

			print("%s: END OF EPOCH %s, log_joint=%s" % (datetime.now(), epoch, self.log_joints[-1]))


if __name__ == '__main__':
	articles, nav_vocab, nav_embeddings, article_navs, ne_vocab, article_nes = \
		preprocess.preprocess_articles(['en', 'es', 'ru'], datetime(1996, 9, 1), datetime(1996, 9, 2))

	training_data = TrainingData(
		articles, nav_vocab, nav_embeddings, article_navs, ne_vocab, article_nes)

	num_nav_topics = 10
	sampler = Sampler("naive", training_data, num_nav_topics=num_nav_topics, 
		nav_topic_mean_prior_mean=np.mean(nav_embeddings, axis=0),
		nav_topic_mean_prior_covariance=np.cov(nav_embeddings, rowvar=False),
		nav_topic_covariance_prior_dof=nav_embeddings.shape[1],
		nav_topic_covariance_prior_scale=3 * nav_embeddings.shape[1] * np.eye(nav_embeddings.shape[1], dtype=np.float64),
		nav_article_topic_proportions_prior_alpha=np.ones(num_nav_topics)
	)
	sampler.run(100)
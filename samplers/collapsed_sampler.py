import numpy as np
from datetime import datetime
from numpy.random import choice
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from scipy.stats import dirichlet, invwishart, multivariate_normal

from preprocess import Article

from utils import (
	CovarianceMatrix, 
	get_kmeans_assignments,
	logsumexp,
	calculate_multivariate_normal_logpdf,
	calculate_multivariate_t_logpdf
)

class CollapsedSampler:
	def __init__(self, modes, training_data, 
		num_nav_topics, nav_topic_mean_prior_means, nav_topic_mean_prior_kappa, 
		nav_topic_covariance_prior_dof, nav_topic_covariance_prior_scale, 
		nav_article_topic_proportions_prior_alpha, nav_initialization,
		num_ne_topics, ne_topic_vocab_prior_alpha, ne_article_topic_proportions_prior_alpha
	):
		self.modes = modes
		self.training_data = training_data

		##
		# noun and verb hyperparameters
		##
		self.num_nav_topics = num_nav_topics
		# multivariate-Gaussian prior on our topic means
		self.nav_topic_mean_prior_means = nav_topic_mean_prior_means
		self.nav_topic_mean_prior_kappa = nav_topic_mean_prior_kappa
		# inverse-Wishart prior on our topic covariances
		self.nav_topic_covariance_prior_dof = nav_topic_covariance_prior_dof
		self.nav_topic_covariance_prior_scale = nav_topic_covariance_prior_scale
		# Dirichlet prior on our article topic proportions
		self.nav_article_topic_proportions_prior_alpha = nav_article_topic_proportions_prior_alpha

		self.nav_initialization = nav_initialization

		##
		# named entity hyperparameters
		##
		self.num_ne_topics = num_ne_topics
		# Dirichlet prior on our named entity vocabulary distribution
		self.ne_topic_vocab_prior_alpha = ne_topic_vocab_prior_alpha
		# Dirichlet prior on our article topic proportions
		self.ne_article_topic_proportions_prior_alpha = ne_article_topic_proportions_prior_alpha


	def initialize_nav_intermediate_values(self):
		# num topics x [((article_nav_id, nav_id) -> idx in nav_ids, nav_ids, # of assignments, sum of embeddings)]
		self.nav_topic_stats = defaultdict(lambda: [{}, [], 0, np.zeros(self.training_data.nav_embeddings.shape[1])])
		# num articles x num topics 
		self.nav_article_topic_counts = np.zeros((len(self.training_data.articles), self.num_nav_topics))

		for article_id in range(len(self.training_data.article_navs)):
			for article_nav_id, nav_id in enumerate(self.training_data.article_navs[article_id]):
				nav_topic_assignment = self.nav_article_nav_assignments[-1][article_id][article_nav_id]

				nav = self.training_data.nav_embeddings[nav_id]
				self.nav_topic_stats[nav_topic_assignment][0][(article_nav_id, nav_id)] = \
					len(self.nav_topic_stats[nav_topic_assignment][1])
				self.nav_topic_stats[nav_topic_assignment][1].append(nav_id)
				self.nav_topic_stats[nav_topic_assignment][2] += 1
				self.nav_topic_stats[nav_topic_assignment][3] += nav
				self.nav_article_topic_counts[article_id][nav_topic_assignment] += 1

		self.nav_topic_kappas = np.full(
			self.num_nav_topics, fill_value=self.nav_topic_mean_prior_kappa)
		self.nav_topic_dofs = np.full(
			self.num_nav_topics, fill_value=self.nav_topic_covariance_prior_dof)
		self.nav_topic_means = [np.zeros(self.training_data.nav_embeddings.shape[1]) for _ in range(self.num_nav_topics)]
		self.nav_topic_scales = [np.copy(self.nav_topic_covariance_prior_scale) for _ in range(self.num_nav_topics)]

		for nav_topic_id in range(self.num_nav_topics):
			article_navs_to_nav_ids, topic_nav_ids, topic_nav_count, topic_nav_sum = self.nav_topic_stats[nav_topic_id]

			self.nav_topic_kappas[nav_topic_id] += topic_nav_count
			self.nav_topic_dofs[nav_topic_id] += topic_nav_count
			self.nav_topic_means[nav_topic_id] = (1.0 / self.nav_topic_kappas[nav_topic_id]) * (
				self.nav_topic_mean_prior_kappa * self.nav_topic_mean_prior_means[nav_topic_id] + topic_nav_sum)

			topic_navs = self.training_data.nav_embeddings[topic_nav_ids]
			topic_nav_mean = (1.0 / topic_nav_count) * topic_nav_sum
			topic_nav_devs = topic_navs - topic_nav_mean
			self.nav_topic_scales[nav_topic_id] += \
				np.einsum('ki,kj->ij', topic_nav_devs, topic_nav_devs) \
				+ (((self.nav_topic_mean_prior_kappa * topic_nav_count) / self.nav_topic_kappas[nav_topic_id]) \
					* ((topic_nav_mean - self.nav_topic_mean_prior_mean) @ ((topic_nav_mean - self.nav_topic_mean_prior_mean).transpose())))


	def sample_nav_article_nav_assignment(self, article_id, article_nav_id, nav_id):
		nav_embedding = self.training_data.nav_embeddings[nav_id]
		old_assignment = self.nav_article_nav_assignments[-1][article_id][article_nav_id]

		assignment_proportions = np.log(
			self.nav_article_topic_counts[article_id] + self.nav_article_topic_proportions_prior_alpha)
		for nav_topic_id in range(self.num_nav_topics):
			mean, kappa, dof, scale = (
				self.nav_topic_means[nav_topic_id],
				self.nav_topic_kappas[nav_topic_id],
				self.nav_topic_dofs[nav_topic_id],
				self.nav_topic_scales[nav_topic_id]
			)
			cov = scale / (dof - nav_embedding.shape[0] + 1)
			assignment_proportions[nav_topic_id] += \
				calculate_multivariate_t_logpdf(
					nav_embedding, mean, ((kappa + 1)/kappa) * cov, dof - nav_embedding.shape[0] + 1
				)
		assignment_proportions = np.nan_to_num(
			assignment_proportions, copy=False, nan=np.nanmin(assignment_proportions))
		assignment_probabilities = np.exp(
			assignment_proportions - logsumexp(assignment_proportions))

		try:
			updated_assignment = choice(self.num_nav_topics, p=assignment_probabilities)
		except:
			print(assignment_proportions.min())
			print(assignment_proportions)
			print(assignment_probabilities)
			raise

		if old_assignment != updated_assignment:
			# update values for old topic assignment

			self.nav_article_topic_counts[article_id][old_assignment] -= 1

			nav_idx = self.nav_topic_stats[old_assignment][0].pop((article_nav_id, nav_id))
			del self.nav_topic_stats[old_assignment][1][nav_idx]
			self.nav_topic_stats[old_assignment][2] -= 1
			self.nav_topic_stats[old_assignment][3] -= nav_embedding

			self.nav_topic_scales[old_assignment] -= \
				(self.nav_topic_kappas[old_assignment]/(self.nav_topic_kappas[old_assignment] - 1)) \
				* ((self.nav_topic_means[old_assignment] - nav_embedding) @ (
					self.nav_topic_means[old_assignment] - nav_embedding).transpose())
			self.nav_topic_means[old_assignment] = (1 / self.nav_topic_kappas[old_assignment] - 1) \
				* ((self.nav_topic_means[old_assignment] * self.nav_topic_kappas[old_assignment]) - nav_embedding)
			self.nav_topic_kappas[old_assignment] -= 1
			self.nav_topic_dofs[old_assignment] -= 1

			# update values for new topic assignment

			self.nav_article_topic_counts[article_id][updated_assignment] += 1

			self.nav_topic_stats[updated_assignment][0][(article_nav_id, nav_id)] = \
				len(self.nav_topic_stats[updated_assignment][1])
			self.nav_topic_stats[updated_assignment][1].append(nav_id)
			self.nav_topic_stats[updated_assignment][2] += 1
			self.nav_topic_stats[updated_assignment][3] += nav_embedding

			self.nav_topic_dofs[updated_assignment] += 1
			self.nav_topic_kappas[updated_assignment] += 1
			self.nav_topic_means[updated_assignment] = (1 / self.nav_topic_kappas[updated_assignment]) \
				* ((self.nav_topic_means[updated_assignment] * (self.nav_topic_kappas[updated_assignment] - 1)) + nav_embedding)
			self.nav_topic_scales[updated_assignment] += \
				(self.nav_topic_kappas[updated_assignment]/(self.nav_topic_kappas[updated_assignment] - 1)) \
				* ((self.nav_topic_means[updated_assignment] - nav_embedding) @ (
					self.nav_topic_means[updated_assignment] - nav_embedding).transpose())
			
		return updated_assignment


	def calculate_topic_nav_logprob(self, nav_id, mean, cov):
		return calculate_multivariate_normal_logpdf(
			self.training_data.nav_embeddings[nav_id], mean, cov
		)	


	def calculate_nav_log_joint(self, means, covs):
		log_joint = 0
		for nav_topic_id in range(self.num_nav_topics):
			log_joint += calculate_multivariate_normal_logpdf(
				means[nav_topic_id], 
				self.nav_topic_mean_prior_mean, 
				covs[nav_topic_id].matrix
			)
			log_joint += invwishart.logpdf(
				covs[nav_topic_id].matrix,
				self.nav_topic_covariance_prior_dof,
				self.nav_topic_covariance_prior_scale
			)

		for article_id in range(len(self.training_data.articles)):
			# log_joint += dirichlet.logpdf(
			# 	self.nav_article_proportions[-1][article_id],
			# 	self.nav_article_topic_proportions_prior_alpha
			# )

			for article_nav_id, nav_id in enumerate(self.training_data.article_navs[article_id]):
				nav_topic_assignment = self.nav_article_nav_assignments[-1][article_id][article_nav_id]
				# log_joint += np.log(self.nav_article_proportions[-1][article_id][nav_topic_assignment])
				log_joint += self.calculate_topic_nav_logprob(
					nav_id, means[nav_topic_assignment], covs[nav_topic_assignment])

		return log_joint


	def calculate_ne_intermediate_values(self):
		# num topics x # nes
		self.topic_ne_counts = np.zeros((self.num_ne_topics, len(self.training_data.ne_vocab)))
		# num articles x num topics 
		self.ne_article_topic_counts = np.zeros((len(self.training_data.articles), self.num_ne_topics))

		for article_id in range(len(self.training_data.articles)):
			for article_ne_id, ne_id in enumerate(self.training_data.article_nes[article_id]):
				ne_topic_assignment = self.ne_article_ne_assignments[-1][article_id][article_ne_id]

				self.topic_ne_counts[ne_topic_assignment][ne_id] += 1
				self.ne_article_topic_counts[article_id][ne_topic_assignment] += 1


	def sample_ne_topic_proportions(self, topic_id):
		return dirichlet.rvs(
			self.ne_topic_vocab_prior_alpha \
			+ self.topic_ne_counts[topic_id]
		)[0]


	def sample_ne_article_proportions(self, article_id):
		return dirichlet.rvs(
			self.ne_article_topic_proportions_prior_alpha \
			+ self.ne_article_topic_counts[article_id]
		)[0]


	def sample_ne_article_assignments(self, article_id):
		ne_article_proportions = self.ne_article_proportions[-1][article_id]
		unnormalized = np.array([
			[ne_article_proportions[topic_id] * self.ne_topic_proportions[-1][topic_id][ne_id] \
				for topic_id in range(self.num_ne_topics)] \
			for ne_id in self.training_data.article_nes[article_id]
		])
		normalized = unnormalized / unnormalized.sum(axis = 1, keepdims=True)
		return [choice(self.num_ne_topics, p=p) for p in normalized]


	def calculate_ne_log_joint(self):
		log_joint = 0
		for ne_topic_id in range(self.num_ne_topics):
			log_joint += dirichlet.logpdf(
				self.ne_topic_proportions[-1][ne_topic_id],
				self.ne_topic_vocab_prior_alpha
			)

		for article_id in range(len(self.training_data.articles)):
			log_joint += dirichlet.logpdf(
				self.ne_article_proportions[-1][article_id],
				self.ne_article_topic_proportions_prior_alpha
			)

			for article_ne_id, ne_id in enumerate(self.training_data.article_nes[article_id]):
				ne_topic_assignment = self.ne_article_ne_assignments[-1][article_id][article_ne_id]
				log_joint += np.log(self.ne_article_proportions[-1][article_id][ne_topic_assignment])
				log_joint += np.log(self.ne_topic_proportions[-1][ne_topic_assignment][ne_id])

		return log_joint


	def run(self, num_iterations):
		if 'nav' in self.modes:
			##
			# SAMPLED VALUES
			##

			# num iterations x num articles x num navs per document
			if self.nav_initialization == 'r':
				self.nav_article_nav_assignments = [[
					[choice(self.num_nav_topics) for _ in self.training_data.article_navs[article_id]] \
					for article_id in range(len(self.training_data.articles))
				]]
			elif self.nav_initialization == 'k':
				assignments, _ = get_kmeans_assignments(
					self.training_data.nav_embeddings, self.num_nav_topics)

				self.nav_article_nav_assignments = [[
					[assignments[nav_id] for nav_id in self.training_data.article_navs[article_id]] \
					for article_id in range(len(self.training_data.articles))
				]]
			else:
				raise Exception("Unsupported initialization: %s" % (self.nav_initialization))

			self.initialize_nav_intermediate_values()

			##
			# MONITORING VALUES
			##

			self.nav_log_joints = []

			print("%s: beginning nav sampling..." % datetime.now())

			for epoch in range(num_iterations):
				print("%s: sampling nav topic assignments" % (datetime.now()))

				updated_nav_article_nav_assignments = []
				for article_id in range(len(self.training_data.articles)):
					if article_id % 1 == 0:
						print("%s: article %s/%s" % (datetime.now(), article_id, len(self.training_data.articles)))
					updated_nav_article_nav_assignments.append([])
					for article_nav_id, nav_id in enumerate(self.training_data.article_navs[article_id]):
						updated_nav_article_nav_assignments[-1].append(self.sample_nav_article_nav_assignment(
							article_id, article_nav_id, nav_id
						))
					if article_id == 2:
						raise Exception("HALT")
				self.nav_article_nav_assignments.append(updated_nav_article_nav_assignments)

				means, covs = [], []
				for nav_topic_id in range(self.num_nav_topics):
					mean, _, dof, scale = (
						self.nav_topic_means[nav_topic_id],
						self.nav_topic_kappas[nav_topic_id],
						self.nav_topic_dofs[nav_topic_id],
						self.nav_topic_scales[nav_topic_id]
					)
					cov = scale / (dof - nav_embedding.shape[0] + 1)
					means.append(mean)
					covs.append(CovarianceMatrix(cov))

				self.nav_log_joints.append(self.calculate_nav_log_joint(means, covs))

				if epoch % 1 == 0:
					for topic_id in range(self.num_nav_topics):
						top_nav_ids = np.argsort(-np.array([
							self.calculate_topic_nav_logprob(nav_id, means[topic_id], covs[topic_id]) \
							for nav_id in range(len(self.training_data.nav_vocab))
						]))[:5]
						print("topic %s: %s" % (
								topic_id, 
								",".join([
									self.training_data.nav_vocab[nav_id][1] for nav_id in top_nav_ids])
							)
						)

				print("%s: nav, END OF EPOCH %s, log_joint=%s" % (datetime.now(), epoch, self.nav_log_joints[-1]))

		if 'ne' in self.modes:
			self.ne_topic_proportions = []
			self.ne_article_proportions = []
			self.ne_article_ne_assignments = [[
				[choice(self.num_ne_topics) for _ in range(len(self.training_data.article_nes[article_id]))] \
				for article_id in range(len(self.training_data.articles))
			]]

			self.ne_log_joints = []

			print("%s: beginning ne sampling..." % datetime.now())

			for epoch in range(num_iterations):
				self.calculate_ne_intermediate_values()

				updated_ne_topic_proportions = []
				for ne_topic_id in range(self.num_ne_topics):
					updated_ne_topic_proportions.append(self.sample_ne_topic_proportions(ne_topic_id))
				self.ne_topic_proportions.append(updated_ne_topic_proportions)

				updated_ne_article_proportions = []
				for article_id in range(len(self.training_data.articles)):
					updated_ne_article_proportions.append(self.sample_ne_article_proportions(article_id))
				self.ne_article_proportions.append(updated_ne_article_proportions)

				updated_ne_article_ne_assignments = []
				for article_id in range(len(self.training_data.articles)):
					if len(self.training_data.article_nes[article_id]) == 0:
						updated_ne_article_ne_assignments.append([])
					else:
						updated_ne_article_ne_assignments.append(self.sample_ne_article_assignments(article_id))

				self.ne_article_ne_assignments.append(updated_ne_article_ne_assignments)

				self.ne_log_joints.append(self.calculate_ne_log_joint())

				print("%s: ne, END OF EPOCH %s, log_joint=%s" % (datetime.now(), epoch, self.ne_log_joints[-1]))

				if epoch % 10 == 0:
					for ne_topic_id in range(self.num_ne_topics):
						top_ne_ids = np.argsort(
							-np.stack(updated_ne_topic_proportions[ne_topic_id]))[:5]
						top_article_ids = np.argsort(
							-np.stack(updated_ne_article_proportions)[:, ne_topic_id])[:5]
						print("topic %s: %s; %s" % (
								ne_topic_id, 
								",".join([
									self.training_data.ne_vocab[ne_id] for ne_id in top_ne_ids]), 
								",".join(map(str, top_article_ids))
							)
						)
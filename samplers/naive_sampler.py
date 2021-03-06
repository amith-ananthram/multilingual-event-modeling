import os
import pickle
import numpy as np
from datetime import datetime
from numpy.random import choice
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from collections import defaultdict, Counter
from scipy.stats import dirichlet, invwishart, multivariate_normal

from utils import (
	CovarianceMatrix, 
	get_kmeans_assignments,
	logsumexp,
	calculate_multivariate_normal_logpdf
)

OUTPUT_DIR = 'output'

class NaiveSampler:
	def __init__(self, variant, modes, num_workers, training_data, 
		num_nav_topics, nav_topic_mean_prior_means, nav_topic_mean_prior_kappa, 
		nav_topic_covariance_prior_dof, nav_topic_covariance_prior_scale, 
		nav_article_topic_proportions_prior_alpha, nav_initialization,
		num_ne_topics, ne_topic_vocab_prior_alpha, ne_article_topic_proportions_prior_alpha
	):
		self.cache = {}
		self.modes = modes
		self.variant = variant
		self.num_workers = num_workers
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


	def calculate_nav_intermediate_values(self):
		# num topics x [(assigned navs, # of assignments, sum of embeddings)]
		self.nav_topic_stats = defaultdict(lambda: [[], 0, np.zeros(self.training_data.nav_embeddings.shape[1])])
		# num articles x num topics 
		self.nav_article_topic_counts = np.zeros((len(self.training_data.articles), self.num_nav_topics))

		for article_id in range(len(self.training_data.article_navs)):
			for article_nav_id, nav_id in enumerate(self.training_data.article_navs[article_id]):
				nav_topic_assignment = self.nav_article_nav_assignments[-1][article_id][article_nav_id]

				nav = self.training_data.nav_embeddings[nav_id]
				self.nav_topic_stats[nav_topic_assignment][0].append(nav)
				self.nav_topic_stats[nav_topic_assignment][1] += 1
				self.nav_topic_stats[nav_topic_assignment][2] += nav
				self.nav_article_topic_counts[article_id][nav_topic_assignment] += 1

		for nav_topic_id in range(self.num_nav_topics):
			if self.nav_topic_stats[nav_topic_id][1] == 0:
				print("empty topic: %s" % nav_topic_id)


	def sample_nav_topic_mean_and_covariance(self, topic_id):
		topic_navs, topic_nav_count, topic_nav_sum = self.nav_topic_stats[topic_id]

		##
		# re-sample the topic covariance
		##

		topic_covariance_dof = self.nav_topic_covariance_prior_dof + topic_nav_count

		# the einsum below produces the sum of the outer products
		# of the deviations of each word vector from the topic mean
		if len(topic_navs) > 0:
			topic_nav_devs = np.stack(topic_navs) - self.nav_topic_means[-1][topic_id] 
			topic_covariance_scale = self.nav_topic_covariance_prior_scale + \
				np.einsum('ki,kj->ij', topic_nav_devs, topic_nav_devs)
		else:
			topic_covariance_scale = self.nav_topic_covariance_prior_scale

		topic_covariance = invwishart.rvs(
			topic_covariance_dof, topic_covariance_scale)

		##
		# re-sample the topic mean
		##

		topic_mean_mean = (
			self.nav_topic_mean_prior_kappa * self.nav_topic_mean_prior_means[topic_id] + topic_nav_sum) \
			/ (self.nav_topic_mean_prior_kappa + topic_nav_count) 
		topic_mean_covariance = (topic_covariance / (
			self.nav_topic_mean_prior_kappa + topic_nav_count))
		topic_mean = multivariate_normal.rvs(
			topic_mean_mean,
			topic_mean_covariance
		)

		return topic_mean, CovarianceMatrix(topic_covariance)


	def sample_nav_article_proportions(self, article_id):
		return dirichlet.rvs(
			self.nav_article_topic_proportions_prior_alpha \
			+ self.nav_article_topic_counts[article_id]
		)[0]


	def calculate_topic_nav_logprob(self, topic_id, nav_id):
		cache_key = ("topic_nav_logprob", topic_id, nav_id)
		if cache_key not in self.cache:
			self.cache[cache_key] = calculate_multivariate_normal_logpdf(
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
				assignment_proportions - logsumexp(assignment_proportions))
			updated_assignments.append(choice(self.num_nav_topics, p=assignment_probabilities))

		return updated_assignments


	def calculate_nav_log_joint(self):
		log_joint = 0
		for nav_topic_id in range(self.num_nav_topics):
			log_joint += calculate_multivariate_normal_logpdf(
				self.nav_topic_means[-1][nav_topic_id], 
				self.nav_topic_mean_prior_means[nav_topic_id], 
				CovarianceMatrix.scaled(
					self.nav_topic_covariances[-1][nav_topic_id], (1/self.nav_topic_mean_prior_kappa))
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
		if len(self.training_data.article_nes[article_id]) == 0:
			return []

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


	def plot_log_joints(self, log_joints, title):
		plt.plot(log_joints)
		plt.ylabel('Log Joint')
		plt.title(title)
		plt.savefig(os.path.join(OUTPUT_DIR, "%s-log-joints.png" % self.variant))


	def run(self, num_iterations):
		if 'nav' in self.modes:
			##
			# SAMPLED VALUES
			##
			# num iterations x num topics x word embedding dim^2
			self.nav_topic_covariances = []
			# num iterations x num articles x num topics
			self.nav_article_proportions = []

			if self.nav_initialization == 'r':
				# num iterations x num topics x word embedding dim
				self.nav_topic_means = [[
					np.random.randn(self.training_data.nav_embeddings.shape[1]) for _ in range(self.num_nav_topics)
				]]
				# num iterations x num articles x num navs per document
				self.nav_article_nav_assignments = [[
					[choice(self.num_nav_topics) for _ in range(len(self.training_data.article_navs[article_id]))]
					for article_id in range(len(self.training_data.articles))
				]]
			elif self.nav_initialization == 'k':
				assignments, means = get_kmeans_assignments(
					self.training_data.nav_embeddings, self.num_nav_topics)

				# num iterations x num topics x word embedding dim
				self.nav_topic_means = [means]
				# num iterations x num articles x num navs per document
				self.nav_article_nav_assignments = [[
					[assignments[nav_id] for nav_id in self.training_data.article_navs[article_id]] \
					for article_id in range(len(self.training_data.articles))
				]]
			else:
				raise Exception("Unsupported initialization: %s" % (self.nav_initialization))

			##
			# MONITORING VALUES
			##

			self.nav_log_joints = []

			print("%s: beginning nav sampling..." % datetime.now())

			with ThreadPool(self.num_workers) as p:
				for epoch in range(num_iterations):
					# general purpose cache for 
					# memoizing expensive ops
					self.cache = {}
					self.calculate_nav_intermediate_values()

					
					print("%s: sampling nav topics" % (datetime.now()))

					# re-sample nav topic means & covariances
					updated_nav_topic_means_and_covariances = p.map(
						self.sample_nav_topic_mean_and_covariance, range(self.num_nav_topics)
					)
					self.nav_topic_means.append(
						[nav_updated_topic_mean for (nav_updated_topic_mean, _) in updated_nav_topic_means_and_covariances])
					self.nav_topic_covariances.append(
						[nav_updated_topic_covariance for (_, nav_updated_topic_covariance) in updated_nav_topic_means_and_covariances])

					print("%s: sampling nav article proportions" % (datetime.now()))

					# re-sample nav article proportions
					self.nav_article_proportions.append(p.map(
						self.sample_nav_article_proportions, 
						range(len(self.training_data.articles))
					))

					print("%s: sampling nav topic assignments" % (datetime.now()))

					# re-sample nav article nav assignments
					self.nav_article_nav_assignments.append(p.map(
						self.sample_nav_article_nav_assignments,
						range(len(self.training_data.articles))
					))

					self.nav_log_joints.append(self.calculate_nav_log_joint())

					if epoch % 1 == 0:
						for topic_id in range(self.num_nav_topics):
							top_nav_ids = np.argsort(-np.array([
								self.calculate_topic_nav_logprob(topic_id, nav_id) \
								for nav_id in range(len(self.training_data.nav_vocab))
							]))[:5]
							top_article_ids = np.argsort(
								-np.stack(self.nav_article_proportions[-1])[:, topic_id])[:5]
							print("topic %s: %s; %s" % (
									topic_id, 
									",".join([
										self.training_data.nav_vocab[nav_id][1] for nav_id in top_nav_ids]), 
									",".join(map(str, top_article_ids))
								)
							)

					print("%s: nav, END OF EPOCH %s, log_joint=%s" % (datetime.now(), epoch, self.nav_log_joints[-1]))

			self.plot_log_joints(self.nav_log_joints, "Nouns & Verbs, Log Joint")

		if 'ne' in self.modes:
			self.ne_topic_proportions = []
			self.ne_article_proportions = []
			self.ne_article_ne_assignments = [[
				[choice(self.num_ne_topics) for _ in range(len(self.training_data.article_nes[article_id]))] \
				for article_id in range(len(self.training_data.articles))
			]]

			self.ne_log_joints = []

			print("%s: beginning ne sampling..." % datetime.now())

			with ThreadPool(self.num_workers) as p:
				for epoch in range(num_iterations):
					self.calculate_ne_intermediate_values()

					self.ne_topic_proportions.append(p.map(
						self.sample_ne_topic_proportions, 
						range(self.num_ne_topics)
					))

					self.ne_article_proportions.append(p.map(
						self.sample_ne_article_proportions, 
						range(len(self.training_data.articles))
					))

					self.ne_article_ne_assignments.append(p.map(
						self.sample_ne_article_assignments, 
						range(len(self.training_data.articles))
					))

					self.ne_log_joints.append(self.calculate_ne_log_joint())

					print("%s: ne, END OF EPOCH %s, log_joint=%s" % (datetime.now(), epoch, self.ne_log_joints[-1]))

					if epoch % 10 == 0:
						missing_nes = Counter()
						articles_by_lang = defaultdict(Counter)
						for article_id in range(len(self.ne_article_proportions[-1])):
							lang = self.training_data.articles[article_id].lang
							if len(self.training_data.article_nes[article_id]) == 0:
								missing_nes[lang] += 1
							else:
								max_topic_id = np.argmax(self.ne_article_proportions[-1][article_id])
								articles_by_lang[max_topic_id][lang] += 1

						print("missing NEs: %s" % (missing_nes))
						for ne_topic_id in range(self.num_ne_topics):
							top_ne_ids = np.argsort(
								-self.ne_topic_proportions[-1][ne_topic_id])[:5]
							top_article_ids = np.argsort(
								-np.stack(self.ne_article_proportions[-1])[:, ne_topic_id])[:5]

							print("topic %s: %s; %s; %s" % (
									ne_topic_id, 
									",".join([
										self.training_data.ne_vocab[ne_id] for ne_id in top_ne_ids]), 
									",".join(map(str, top_article_ids)),
									",".join(map(lambda lang_count: "%s-%s" % lang_count, 
										articles_by_lang[ne_topic_id].items()))
								)
							)

			self.plot_log_joints(self.ne_log_joints, "Named Entities, Log Joint")

			with open(os.path.join(OUTPUT_DIR, '%s-topics.pkl' % (self.variant)), 'wb') as f:
				pickle.dump(self.ne_topic_proportions, f)

			with open(os.path.join(OUTPUT_DIR, '%s-proportions.pkl' % (self.variant)), 'wb') as f:
				pickle.dump(self.ne_article_proportions, f)

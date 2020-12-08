import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

import math
import pyro
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
import pyro.distributions as dist

from sklearn.neighbors import NearestNeighbors

import preprocess

NUM_TOPICS = 10

# produce variational parameters of approximate posterior distribution
# of document proportions from a bag of words representation of the document
class ProportionEncoder(nn.Module):
	def __init__(self, dropout, vocab_size, hidden, num_topics):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.fc1 = nn.Linear(vocab_size, hidden)
		self.fc2 = nn.Linear(hidden, hidden)

		# produce mean of log-normal distribution
		# that approximates proportion posterior
		self.prop_mu = nn.Linear(hidden, num_topics)
		self.bn_prop_mu = nn.BatchNorm1d(num_topics)

		# produce diagonal of covariance matrix
		# of the log-normal distribution that 
		# approximates proportion posterior
		self.prop_sigma = nn.Linear(hidden, num_topics)
		self.bn_prop_sigma = nn.BatchNorm1d(num_topics)

	# bows: batch_size x words in documents
	def forward(self, bows):
		h = F.softplus(self.fc1(bows))
		h = self.dropout(F.softplus(self.fc2(h)))

		prop_mu = self.bn_prop_mu(self.prop_mu(h))
		prop_sigma = (0.5 * self.bn_prop_sigma(self.prop_sigma(h))).exp()

		return prop_mu, prop_sigma

# produce generative samples from our joint distribution conditional
# on the sample oof latent variables from our approximate posterior distribution
class TopicDecoder(nn.Module):
	def __init__(self, dropout, num_topics, embedding_dim):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

		# produce mean of multivariate normal distribution
		# to calculate the likelihood of a given embedding
		self.topics_mu = nn.Linear(num_topics, embedding_dim)
		self.bn_topics_mu = nn.BatchNorm1d(embedding_dim)

		# produce diagonal of covariance matrix
		# of the multivariate normal distribution 
		# to calculate the likelihood of a given embedding
		self.topics_sigma = nn.Linear(num_topics, embedding_dim)
		self.bn_topics_sigma = nn.BatchNorm1d(embedding_dim)

	# proportions: batch_size x num_topics
	def forward(self, proportions):
		topics_mu = self.bn_topics_mu(self.topics_mu(proportions))
		topics_sigma = (0.5 * self.bn_topics_sigma(self.topics_sigma(proportions))).exp()

		return topics_mu, topics_sigma


class NavEmbeddingLDA(nn.Module):
	def __init__(self, nav_vocabulary, nav_embeddings, article_nav_bows):
		super().__init__()
		self.nav_vocabulary = nav_vocabulary
		self.nav_embeddings = nav_embeddings
		self.article_navs = [
			[nav_id for (nav_id, count) in enumerate(article_nav_bows[document_id]) if count > 0] \
			for document_id in range(len(article_nav_bows))
		]

		self.prop_inference_net = ProportionEncoder(
			0.2, len(nav_vocabulary), 100, NUM_TOPICS)
		self.topic_recognition_net = TopicDecoder(
			0.2, NUM_TOPICS, self.nav_embeddings.shape[1])

	def guide(self, bows, _):
		pyro.module("prop_inference_net", self.prop_inference_net)
		with pyro.plate("documents", bows.shape[0]):
			prop_mu, prop_sigma = self.prop_inference_net(bows)
			props = pyro.sample(
				"theta", dist.LogNormal(prop_mu, prop_sigma).to_event(1))

	# bows: batch_size x vocabulary size
	def model(self, bows, document_ids):
		pyro.module("topic_recognition_net", self.topic_recognition_net)
		with pyro.plate("documents", bows.shape[0]):
			# instead of a Dirichlet prior, we use a log-normal distribution
			prop_mu = bows.new_zeros((bows.shape[0], NUM_TOPICS))
			prop_sigma = bows.new_ones((bows.shape[0], NUM_TOPICS))
			props = pyro.sample(
				"theta", dist.LogNormal(prop_mu, prop_sigma).to_event(1))

			topics_mu, topics_sigma = self.topic_recognition_net(props)

			for document_index in range(bows.shape[0]):
				document_id = document_ids[document_index]
				for nav_id in pyro.plate("navs_{}".format(document_id), len(self.article_navs[document_id])):
					pyro.sample(
						"nav_{}_{}".format(document_id, nav_id), 
						dist.MultivariateNormal(
							topics_mu[document_index], 
							scale_tril=torch.diag(topics_sigma[document_index]).double()
						).to_event(0),
						obs=self.nav_embeddings[self.article_navs[document_id][nav_id]]	
					)

	def get_topics(self):
		return self.topic_recognition_net.topics_mu.weight.cpu().detach().transpose(1, 0)

if __name__ == '__main__':
	articles, nav_vocabulary, nav_embeddings, article_nav_bows = \
		preprocess.preprocess_articles(['en', 'es', 'ru'], datetime(1996, 9, 1), datetime(1996, 10, 1))

	seed = 0
	torch.manual_seed(seed)
	pyro.set_rng_seed(seed)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	article_nav_bows = torch.tensor(article_nav_bows[0:100]).float().to(device)
	batch_size = 32
	learning_rate = 1e-3
	num_epochs = 50

	pyro.clear_param_store()

	nav_embeddings = torch.tensor(nav_embeddings)
	navLDA = NavEmbeddingLDA(
		nav_vocabulary=nav_vocabulary,
		nav_embeddings=nav_embeddings,
		article_nav_bows=article_nav_bows
	)
	navLDA.to(device)

	optimizer = pyro.optim.Adam({"lr": learning_rate})
	svi = SVI(navLDA.model, navLDA.guide, optimizer, loss=TraceMeanField_ELBO())

	losses = []
	num_batches = int(math.ceil(article_nav_bows.shape[0] / batch_size))
	for epoch in range(num_epochs):
		# shuffle data?

		epoch_loss = 0.0
		for batch_id in range(num_batches):
			article_ids = list(range(batch_id * batch_size, (batch_id + 1) * batch_size))
			losses.append(svi.step(article_nav_bows[
				batch_id * batch_size:(batch_id + 1) * batch_size, :], article_ids))

			epoch_loss += losses[-1]
			if batch_id % 100 == 0:
				print("%s: EPOCH %s, BATCH %s/%s: loss=%s" % (datetime.now(), epoch, batch_id, num_batches, losses[-1]))

		print("%s: END OF EPOCH %s, loss=%s" % (datetime.now(), epoch, epoch_loss))

	topic_centers = navLDA.get_topics()

	embedding_neighbors = NearestNeighbors(n_neighbors=10)
	embedding_neighbors.fit(nav_embeddings)
	topic_neighborhoods = embedding_neighbors.kneighbors(
		topic_centers.numpy(), n_neighbors=10, return_distance=False)

	for topic_id, topic_neighborhood in enumerate(topic_neighborhoods):
		print(topic_id)
		for neighbor_index in topic_neighborhood:
			print(nav_vocabulary[neighbor_index])
	


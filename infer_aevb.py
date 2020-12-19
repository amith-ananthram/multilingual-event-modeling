import argparse

import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

import math
import pyro
import torch
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
import pyro.distributions as dist

from sklearn.neighbors import NearestNeighbors

from preprocess import Article, preprocess_articles

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
# on the sample of latent variables from our approximate posterior distribution
class NavTopicDecoder(nn.Module):
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


# produce generative samples from our joint distribution conditional
# on the sample of latent variables from our approximate posterior distribution
class NeTopicDecoder(nn.Module):
	def __init__(self, dropout, num_topics, vocab_size):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

		# produce categorical distribution over named entities
		self.topics_beta = nn.Linear(num_topics, vocab_size)
		self.bn_topics_beta = nn.BatchNorm1d(vocab_size)

	# proportions: batch_size x num_topics
	def forward(self, proportions):
		return F.softmax(
			self.bn_topics_beta(self.topics_beta(self.dropout(proportions))), dim=1)


class NavEmbeddingLDA(nn.Module):
	def __init__(self, device, nav_topics, training_data):
		super().__init__()
		self.device = device
		self.nav_topics = nav_topics
		self.nav_vocabulary = training_data.nav_vocab
		self.article_navs = training_data.article_navs

		self.prop_inference_net = ProportionEncoder(
			0.2, len(self.nav_vocabulary), 100, self.nav_topics)
		self.topic_recognition_net = NavTopicDecoder(
			0.2, self.nav_topics, training_data.nav_embeddings.shape[1])

	def guide(self, bows, _, __):
		pyro.module("prop_inference_net", self.prop_inference_net)
		with pyro.plate("articles", bows.shape[0]):
			prop_mu, prop_sigma = self.prop_inference_net(bows)
			props = pyro.sample(
				"theta", dist.LogNormal(prop_mu, prop_sigma).to_event(1))

	# bows: batch_size x vocabulary size
	def model(self, bows, embeddings, article_ids):
		pyro.module("topic_recognition_net", self.topic_recognition_net)
		with pyro.plate("articles", bows.shape[0]):
			# instead of a Dirichlet prior, we use a log-normal distribution
			prop_mu = bows.new_zeros((bows.shape[0], self.nav_topics))
			prop_sigma = bows.new_ones((bows.shape[0], self.nav_topics))
			props = pyro.sample(
				"theta", dist.LogNormal(prop_mu, prop_sigma).to_event(1))

			topics_mu, topics_sigma = self.topic_recognition_net(props)

			for batch_article_id, article_id in enumerate(article_ids):
				nav_embeddings = torch.tensor(embeddings[self.article_navs[article_id]]).to(device)
				for article_nav_id in pyro.plate("navs_{}".format(article_id), len(self.article_navs[article_id])):
					pyro.sample(
						"nav_{}_{}".format(article_id, article_nav_id), 
						dist.MultivariateNormal(
							topics_mu[batch_article_id], 
							scale_tril=torch.diag(topics_sigma[batch_article_id])
						).to_event(0),
						obs=nav_embeddings[article_nav_id]	
					)

	def get_topics(self):
		return self.topic_recognition_net.topics_mu.weight.cpu().detach().transpose(1, 0)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Estimate posterior via Gibbs sampling.')
	parser.add_argument('--modes', dest='modes', help='nouns and verbs (nav), named entities (ne)', default='nav')
	parser.add_argument('--data-start-date', dest='data_start_date', help='YYYYMMDD')
	parser.add_argument('--data-end-date', dest='data_end_date', help='YYYYMMDD')
	parser.add_argument('--data-disallow-repeats', dest='data_disallow_repeats', action='store_true', default=False)
	parser.add_argument('--num-nav-topics', dest='num_nav_topics', default=10)
	parser.add_argument('--nav-article-proportions-prior-alpha', dest='nav_article_proportions_prior_alpha', default=1)
	parser.add_argument('--num-ne-topics', dest='num_ne_topics', default=10)
	parser.add_argument('--ne-topic-prior-alpha', dest='ne_topic_prior_alpha', default=1)
	parser.add_argument('--lr', dest='lr', default=1e-3)
	parser.add_argument('--batch_size', dest='batch_size', default=32)
	parser.add_argument('--num-epochs', dest='num_epochs', default=50)
	args = parser.parse_args()

	modes = args.modes.split(',')
	data_start_date = datetime.strptime(args.data_start_date, '%Y%m%d')
	data_end_date = datetime.strptime(args.data_end_date, '%Y%m%d')
	training_data = preprocess_articles(
		['en', 'es', 'ru'], data_start_date, data_end_date, disallow_repeats=args.data_disallow_repeats)

	training_data.articles = training_data.articles[0:100]
	training_data.article_navs = training_data.article_navs[0:100]

	article_nav_bows = np.zeros(
		(len(training_data.articles), len(training_data.nav_vocab)), dtype=np.float32)
	for article_id in range(len(training_data.article_navs)):
		for nav_id in training_data.article_navs[article_id]:
			article_nav_bows[article_id][nav_id] += 1

	print("loaded %s articles for %s-%s" % (len(training_data.articles), data_start_date, data_end_date))

	seed = 0
	torch.manual_seed(seed)
	pyro.set_rng_seed(seed)
	pyro.clear_param_store()
	torch.set_default_dtype(torch.float32)

	batch_size = int(args.batch_size)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	if 'nav'in modes:
		nav_topics = int(args.num_nav_topics)
		navLDA = NavEmbeddingLDA(device, nav_topics, training_data).to(device)

		optimizer = pyro.optim.Adam({"lr": float(args.lr)})
		svi = SVI(navLDA.model, navLDA.guide, optimizer, loss=TraceMeanField_ELBO())

		losses = []
		num_batches = int(math.ceil(article_nav_bows.shape[0] / batch_size))
		for epoch in range(int(args.num_epochs)):
			article_ids = np.array((range(article_nav_bows.shape[0])))
			np.random.shuffle(article_ids)

			epoch_loss = 0.0
			for batch_id in range(num_batches):
				batch_article_ids = article_ids[
					list(range(batch_id * batch_size, (batch_id + 1) * batch_size))]
				batch = torch.tensor(article_nav_bows[batch_article_ids]).to(device)
				losses.append(svi.step(batch, training_data.nav_embeddings, batch_article_ids))

				epoch_loss += losses[-1]
				if batch_id % 10 == 0:
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
	


import argparse
import numpy as np
from datetime import datetime

from samplers import NaiveSampler, CollapsedSampler
from preprocess import Article, preprocess_articles

# figure out different priors
# refactor
	# support different modes (nav only, ne only)
	# support different intializers
	# support different convergence checks
	# save samples and diagnostic
# analysis: cluster by date and then semantics and named entities
	# how many articles from each language are in each cluster?
	# what are their titles?
# experiment with AEVB
# posterior predictive checks

# 3) re-factor, write tests for different methods
# 4) figure out different priors / initilizations
# 5) save samples and diagnostics, come up with analysis framework 
# 6) kick off AEVB comparable 
# 7) expanded model:
# 3) can we do something w/ dates and/or titles
#	- read through dynamic topic modeling paper (can just write about it in the thing)

# writeup:
#	- mention speed up w Cholesky, derivation


if __name__ == '__main__':
	training_data = preprocess_articles(
		['en', 'es', 'ru'], datetime(1996, 9, 1), datetime(1996, 9, 2))

	num_nav_topics = 10
	num_ne_topics = 50
	# sampler = NaiveSampler(1, training_data, num_nav_topics=num_nav_topics, 
	# 	nav_topic_mean_prior_mean=np.zeros(training_data.nav_embeddings.shape[1]),#np.mean(training_data.nav_embeddings, axis=0),
	# 	nav_topic_mean_prior_covariance=np.cov(training_data.nav_embeddings, rowvar=False),
	# 	nav_topic_covariance_prior_dof=training_data.nav_embeddings.shape[1] + 1,
	# 	nav_topic_covariance_prior_scale=3 * training_data.nav_embeddings.shape[1] * np.eye(training_data.nav_embeddings.shape[1], dtype=np.float64),
	# 	nav_article_topic_proportions_prior_alpha=np.ones(num_nav_topics),
	# 	num_ne_topics=num_ne_topics,
	# 	ne_topic_vocab_prior_alpha=np.ones(len(training_data.ne_vocab)),
	# 	ne_article_topic_proportions_prior_alpha=np.ones(num_ne_topics)
	# )
	sampler = CollapsedSampler(training_data, num_nav_topics=num_nav_topics, 
		nav_topic_mean_prior_mean=np.zeros(training_data.nav_embeddings.shape[1]),
		nav_topic_mean_prior_covariance_kappa=1,
		nav_topic_covariance_prior_dof=training_data.nav_embeddings.shape[1] + 1,
		nav_topic_covariance_prior_scale=3 * training_data.nav_embeddings.shape[1] * np.eye(training_data.nav_embeddings.shape[1], dtype=np.float64),
		nav_article_topic_proportions_prior_alpha=np.ones(num_nav_topics),
		num_ne_topics=num_ne_topics,
		ne_topic_vocab_prior_alpha=np.ones(len(training_data.ne_vocab)),
		ne_article_topic_proportions_prior_alpha=np.ones(num_ne_topics)
	)
	sampler.run(100)
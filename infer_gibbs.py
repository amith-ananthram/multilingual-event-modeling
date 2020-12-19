import cProfile
import argparse
import numpy as np
from datetime import datetime

import utils
from preprocess import Article, preprocess_articles
from samplers import NaiveSampler, CollapsedSampler, CollapsedCholeskySampler

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Estimate posterior via Gibbs sampling.')
	parser.add_argument('--modes', dest='modes', help='nouns and verbs (nav), named entities (ne)', default='nav')
	parser.add_argument('--sampler', dest='sampler', help='[n]aive, [c]ollapsed or [ch]olesky')
	parser.add_argument('--num-threads', dest='num_threads', help='Parallelization for naive sampler', default=2)
	parser.add_argument('--data-start-date', dest='data_start_date', help='YYYYMMDD')
	parser.add_argument('--data-end-date', dest='data_end_date', help='YYYYMMDD')
	parser.add_argument('--data-disallow-repeats', dest='data_disallow_repeats', action='store_true', default=False)
	parser.add_argument('--num-nav-topics', dest='num_nav_topics', default=10)
	parser.add_argument('--nav-topic-prior-mean', dest='nav_topic_prior_mean', help='[z]ero, [em]bedding mean, [k]-means')
	parser.add_argument('--nav-topic-prior-kappa', dest='nav_topic_prior_kappa', default=1)
	parser.add_argument('--nav-topic-prior-dof-shift', dest='nav_topic_prior_dof_shift', default=1)
	parser.add_argument('--nav-topic-prior-scale-factor', dest='nav_topic_prior_scale_factor', default=3)
	parser.add_argument('--nav-article-proportions-prior-alpha', dest='nav_article_proportions_prior_alpha', default=1)
	parser.add_argument('--nav-initialization', dest='nav_initialization', help='[r]andom, [k]-means')
	parser.add_argument('--num-ne-topics', dest='num_ne_topics', default=10)
	parser.add_argument('--ne-topic-prior-alpha', dest='ne_topic_prior_alpha', default=1)
	parser.add_argument('--ne-article-proportions-prior-alpha', dest='ne_article_proportions_prior_alpha', default=0.1)
	parser.add_argument('--num-iterations', dest='num_iterations', default=10)
	parser.add_argument('--profile', dest='profile', action='store_true', default=False)
	args = parser.parse_args()

	modes = args.modes.split(',')
	data_start_date = datetime.strptime(args.data_start_date, '%Y%m%d')
	data_end_date = datetime.strptime(args.data_end_date, '%Y%m%d')
	training_data = preprocess_articles(
		['en', 'es', 'ru'], data_start_date, data_end_date, disallow_repeats=args.data_disallow_repeats)

	##
	# noun and verb hyperparameters
	##

	num_nav_topics = int(args.num_nav_topics)
	if args.nav_topic_prior_mean == 'z':
		nav_topic_mean_prior_means = [
			np.zeros(training_data.nav_embeddings.shape[1]) for _ in range(num_nav_topics)
		]
	elif args.nav_topic_prior_mean == 'em':
		embedding_mean = np.mean(training_data.nav_embeddings, axis=0) 
		nav_topic_mean_prior_means = [embedding_mean for _ in range(num_nav_topics)]
	elif args.nav_topic_prior_mean == 'k':
		_, nav_topic_mean_prior_means = utils.get_kmeans_assignments(
			training_data.nav_embeddings, num_nav_topics)
	else:
		raise Exception('Unsupported nav_topic_mean_prior_mean: %s' % (args.nav_topic_prior_mean))
	nav_topic_mean_prior_kappa = float(args.nav_topic_prior_kappa)
	nav_topic_covariance_prior_dof = training_data.nav_embeddings.shape[1] + int(args.nav_topic_prior_dof_shift)
	nav_topic_covariance_prior_scale = int(args.nav_topic_prior_scale_factor) \
		* training_data.nav_embeddings.shape[1] * np.eye(training_data.nav_embeddings.shape[1], dtype=np.float64)

	nav_article_topic_proportions_prior_alpha = np.full(
		num_nav_topics, fill_value=float(args.nav_article_proportions_prior_alpha))

	##
	# named entity hyperparameters
	##

	num_ne_topics = int(args.num_ne_topics)
	ne_topic_prior_alpha = np.full(
		len(training_data.ne_vocab), fill_value=float(args.ne_topic_prior_alpha))
	ne_article_proportions_prior_alpha = np.full(
		num_ne_topics, fill_value=float(args.nav_article_proportions_prior_alpha))

	if args.sampler == 'n':
		sampler = NaiveSampler(
			modes, int(args.num_threads), training_data, num_nav_topics=num_nav_topics, 
			nav_topic_mean_prior_means=nav_topic_mean_prior_means,
			nav_topic_mean_prior_kappa=nav_topic_mean_prior_kappa,
			nav_topic_covariance_prior_dof=nav_topic_covariance_prior_dof,
			nav_topic_covariance_prior_scale=nav_topic_covariance_prior_scale,
			nav_article_topic_proportions_prior_alpha=nav_article_topic_proportions_prior_alpha,
			nav_initialization=args.nav_initialization,
			num_ne_topics=num_ne_topics,
			ne_topic_vocab_prior_alpha=ne_topic_prior_alpha,
			ne_article_topic_proportions_prior_alpha=ne_article_proportions_prior_alpha
		)
	elif args.sampler == 'c':
		sampler = CollapsedSampler(
			modes, training_data, num_nav_topics=num_nav_topics, 
			nav_topic_mean_prior_means=nav_topic_mean_prior_means,
			nav_topic_mean_prior_kappa=nav_topic_mean_prior_kappa,
			nav_topic_covariance_prior_dof=nav_topic_covariance_prior_dof,
			nav_topic_covariance_prior_scale=nav_topic_covariance_prior_scale,
			nav_article_topic_proportions_prior_alpha=nav_article_topic_proportions_prior_alpha,
			nav_initialization=args.nav_initialization,
			num_ne_topics=num_ne_topics,
			ne_topic_vocab_prior_alpha=ne_topic_prior_alpha,
			ne_article_topic_proportions_prior_alpha=ne_article_proportions_prior_alpha
		)
	elif args.sampler == 'ch':
		raise Exception("Unimplemented!")
	else:
		raise Exception("Unsupported sampler: %s" % (args.sampler))
	
	num_iterations = int(args.num_iterations)
	if args.profile:
		cProfile.run("sampler.run(%s)" % (num_iterations))
	else:
		sampler.run(num_iterations)
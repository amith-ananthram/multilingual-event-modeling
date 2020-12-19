import argparse
from datetime import datetime
from numpy.random import choice

import tomotopy as tp
from gaussianlda import GaussianLDAAliasTrainer

import utils
from preprocess import Article, preprocess_articles

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Estimate posterior via Gibbs sampling.')
	parser.add_argument('--modes', dest='modes', help='nouns and verbs (nav), named entities (ne)', default='nav')
	parser.add_argument('--data-start-date', dest='data_start_date', help='YYYYMMDD')
	parser.add_argument('--data-end-date', dest='data_end_date', help='YYYYMMDD')
	parser.add_argument('--data-disallow-repeats', dest='data_disallow_repeats', action='store_true', default=False)
	parser.add_argument('--num-nav-topics', dest='num_nav_topics', default=10)
	parser.add_argument('--nav-topic-prior-kappa', dest='nav_topic_prior_kappa', default=1)
	parser.add_argument('--nav-article-proportions-prior-alpha', dest='nav_article_proportions_prior_alpha', default=1)
	parser.add_argument('--nav-initialization', dest='nav_initialization', help='[r]andom, [k]-means')
	parser.add_argument('--num-ne-topics', dest='num_ne_topics', default=10)
	parser.add_argument('--ne-topic-prior-alpha', dest='ne_topic_prior_alpha', default=1)
	parser.add_argument('--ne-article-proportions-prior-alpha', dest='ne_article_proportions_prior_alpha', default=0.1)
	parser.add_argument('--num-iterations', dest='num_iterations', default=10)
	args = parser.parse_args()

	modes = args.modes.split(',')
	data_start_date = datetime.strptime(args.data_start_date, '%Y%m%d')
	data_end_date = datetime.strptime(args.data_end_date, '%Y%m%d')
	training_data = preprocess_articles(
		['en', 'es', 'ru'], data_start_date, data_end_date, disallow_repeats=args.data_disallow_repeats)

	if 'nav' in modes:
		vocab = training_data.nav_vocab
		embeddings = training_data.nav_embeddings
		corpus = training_data.article_navs

		num_nav_topics = int(args.num_nav_topics)
		topics_kappa = float(args.nav_topic_prior_kappa)
		proportions_alpha = float(args.nav_article_proportions_prior_alpha)

		if args.nav_initialization == 'r':
			initializer = lambda article_id, nav_ids: [choice(num_nav_topics) for _ in range(len(nav_ids))]
		elif args.nav_initialization == 'k':
			kmeans_assignments, _ = utils.get_kmeans_assignments(embeddings, num_nav_topics)
			initializer = lambda article_id, nav_ids: [kmeans_assignments[nav_id] for nav_id in nav_ids]
		else:
			raise Exception("Unsupported initialization: %s" % args.nav_initialization)

		trainer = GaussianLDAAliasTrainer(
			corpus, embeddings, vocab, num_nav_topics,
			alpha=proportions_alpha, kappa=topics_kappa
		)
		trainer.sample(int(args.num_iterations))

		print(trainer.format_topics())

		# {}: unused
		# 1: ('en', 'buy') (2.24e-01) ('es', 'comprar') (9.48e-02) ('en', 'investment') (8.58e-02) ('es', 'incrementar') (3.08e-02) ('en', 'government') (2.84e-02) ('en', 'expect') (2.68e-02) ('es', 'aumentar') (2.40e-02) ('en', 'sell') (1.83e-02) ('es', 'implicar') (1.55e-02) ('es', 'disminuir') (1.46e-02)
		# {}: unused
		# {}: unused
		# {}: unused
		# {}: unused
		# {}: unused
		# {}: unused
		# {}: unused
		# {}: unused
		
	if 'ne' in modes:
		raise Exception("Unimplemented!")
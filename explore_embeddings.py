import argparse
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from operator import itemgetter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from preprocess import Article, preprocess_articles

LANG_TO_COLOR_COL = {'en': 0, 'es': 1, 'ru': 2}

def plot_embeddings(embeddings, lang_to_nav_ids, weights, title):
	fig, ax = plt.subplots()
	for lang in ['en', 'es', 'ru']:
		nav_embeddings = list(lang_to_nav_ids[lang])
		colors = np.zeros((len(nav_embeddings), 4))
		colors[:, LANG_TO_COLOR_COL[lang]] = 1
		colors[:, 3] = list(map(
			lambda nav_id: 0.2 + weights[lang][nav_id] * 0.8, 
			nav_embeddings
		))
		plt.scatter(
			embeddings[nav_embeddings, 0], 
			embeddings[nav_embeddings, 1], 
			c=colors, label=lang
		)
	ax.legend()
	plt.title(title)
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Explore word embeddings.')
	parser.add_argument('--data-start-date', dest='data_start_date', help='YYYYMMDD')
	parser.add_argument('--data-end-date', dest='data_end_date', help='YYYYMMDD')
	args = parser.parse_args()

	data_start_date = datetime.strptime(args.data_start_date, '%Y%m%d')
	data_end_date = datetime.strptime(args.data_end_date, '%Y%m%d')
	training_data = preprocess_articles(
		['en', 'es', 'ru'], data_start_date, data_end_date)

	nav_article_counts = defaultdict(Counter)
	for article_id, article in enumerate(training_data.articles):
		for nav_id in set(training_data.article_navs[article_id]):
			nav_article_counts[article.lang][nav_id] += 1

	nav_article_counts_sorted = {}
	for lang in nav_article_counts.keys():
		nav_article_counts_sorted[lang] = list(
			map(itemgetter(0), sorted(nav_article_counts[lang].items(), key=itemgetter(1))))

	nav_ranks_by_lang = defaultdict(dict)
	for lang in nav_article_counts_sorted:
		for nav_id in range(len(training_data.nav_vocab)):
			if nav_id not in nav_article_counts_sorted[lang]:
				nav_ranks_by_lang[lang][nav_id] = 0
			else:
				nav_ranks_by_lang[lang][nav_id] = \
					(len(nav_article_counts_sorted[lang]) - nav_article_counts_sorted[lang].index(nav_id)) / \
					len(nav_article_counts_sorted[lang]) 

	nav_embeddings = training_data.nav_embeddings
	embedding_mean = np.mean(training_data.nav_embeddings, axis=0)
	closest_to_mean = list(
		sorted(
			range(nav_embeddings.shape[0]),
			key=lambda i: np.linalg.norm(nav_embeddings[i] - embedding_mean)
		)
	)
	print("Closest to mean: %s" % (', '.join(
		map(lambda i: '%s-%s' % (training_data.nav_vocab[i]), closest_to_mean[0:10]))))

	# Closest to mean: es-hacerlo, es-condicionar, es-anticiparse, es-cuestionamiento, es-comprometer, 
	# es-preocuparse, es-insistir, es-implicar, es-privilegiar, es-justificación

	lang_to_nav_ids = defaultdict(list)
	for nav_id, (lang, nav) in enumerate(training_data.nav_vocab):
		lang_to_nav_ids[lang].append(nav_id)

	pca_embeddings = PCA(n_components=2).fit_transform(nav_embeddings)
	plot_embeddings(pca_embeddings, lang_to_nav_ids, nav_ranks_by_lang, "Noun and Verb Embeddings (PCA)")

	tsne_embeddings = TSNE(n_components=2).fit_transform(nav_embeddings)
	plot_embeddings(tsne_embeddings, lang_to_nav_ids, nav_ranks_by_lang, "Noun and Verb Embeddings (TSNE)")

	nav_means = np.zeros((len(training_data.articles), training_data.nav_embeddings.shape[1]))
	for article_id in range(len(training_data.article_navs)):
		article_navs = set(training_data.article_navs[article_id])
		for nav_id in article_navs:
			nav_means[article_id] += (1 / len(article_navs)) * training_data.nav_embeddings[nav_id]

	nav_mean_mean = np.mean(nav_means)
	closest_to_mean_mean = list(
		sorted(
			range(training_data.nav_embeddings.shape[0]),
			key=lambda i: np.linalg.norm(training_data.nav_embeddings[i] - nav_mean_mean)
		)
	)
	print("Closest to mean: %s" % (', '.join(
		map(lambda i: '%s-%s' % (training_data.nav_vocab[i]), closest_to_mean_mean[0:10]))))

	# es-milliones, en-titlist, es-elenco, en-book, en-outstanding, es-compatriota, en-revolve, en-chalet, es-crítico, en-def

import argparse
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from operator import itemgetter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from preprocess import Article, preprocess_articles

LANG_TO_COLOR = {'en': '#1f77b4', 'es': '#ff7f0e', 'ru': '#2ca02c'}

def plot_embeddings(embeddings, title):
	fig, ax = plt.subplots()
	for lang in ['en', 'es', 'ru']:
		article_ids = np.array(lang_to_article_ids[lang])
		plt.scatter(
			embeddings[article_ids, 0], embeddings[article_ids, 1], 
			c=LANG_TO_COLOR[lang], label=lang
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

	lang_to_article_ids = defaultdict(list)
	for article_id, article in enumerate(training_data.articles):
		lang_to_article_ids[article.lang].append(article_id)

	pca_embeddings = PCA(n_components=2).fit_transform(nav_embeddings)
	plot_embeddings(pca_embeddings, "Noun and Verb Embeddings (PCA)")

	tsne_embeddings = TSNE(n_components=2).fit_transform(nav_embeddings)
	plot_embeddings(tsne_embeddings, "Noun and Verb Embeddings (TSNE)")
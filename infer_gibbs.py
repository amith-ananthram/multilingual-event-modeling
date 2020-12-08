import numpy as np
from scipy import stats

import preprocess

class TrainingData:
	def __init__(self, articles, noun_and_verb_vocabulary, noun_and_verb_embeddings, article_noun_and_verb_counts):
		self.articles = articles 
		self.noun_and_verb_vocabulary = noun_and_verb_vocabulary
		self.noun_and_verb_embeddings = noun_and_verb_embeddings
		self.article_noun_and_verb_counts = article_noun_and_verb_counts
		self.article_noun_and_verb_vocabularies = [
			[nav_id if count > 0 for nav_id, count in enumerate(article_noun_and_verb_counts[article_id])]
			for article_id in range(article_noun_and_verb_counts.shape[0])
		]

def sample_nav_topic_mean_and_variance(args, nav_topic_id, old_nav_topic_covariance, old_nav_article_word_assignments, data):
	num_navs_in_topic = 0
	summed_nav_in_topic = np.zeros(data.noun_and_verb_embeddings[0].shape)
	for article_id in range(len(old_nav_article_word_assignments)):
		for nav_id, old_nav_topic_id in zip(data.article_noun_and_verb_vocabularies[article_id], old_nav_article_word_assignments[article_id]):
			if old_nav_topic_id == nav_topic_id:
				num_navs_in_topic += data.article_noun_and_verb_counts[article_id][nav_id]
				summed_nav_in_topic += data.article_noun_and_verb_counts[article_id][nav_id] * data.noun_and_verb_embeddings[nav_id]

	nav_mean_covariance_inv = args.nav_mean_prior_covariance_inv + num_navs_in_topic * old_nav_topic_covariance_inv
	nav_mean_covariance = np.linalg.inv(nav_mean_covariance_inv)
	nav_mean_mean = np.matmul(
		nav_mean_covariance_inv,
		np.matmul(args.nav_mean_prior_covariance_inv, args.nav_mean_prior_mean) +	# cache
		np.matmul(old_nav_topic_covariance_inv, summed_nav_in_topic)
	)

	updated_nav_mean = np.random.multivariate_normal(nav_mean_mean, nav_mean_covariance)

	nav_covariance_dof = num_navs_in_topic + args.nav_covariance_prior_dof
	nav_covariance_scale = args.nav_covariance_prior_scale
	for article_id in range(len(old_nav_article_word_assignments)):
		for nav_id, old_nav_topic_id in zip(data.article_noun_and_verb_vocabularies[article_id], old_nav_article_word_assignments[article_id]):
			if old_nav_topic_id == topic_id:
				nav_mean_zero_embedding = data.noun_and_verb_embeddings[nav_id] - updated_nav_mean,
				nav_covariance_scale += data.article_noun_and_verb_counts[article_id][nav_id] * np.matmul(
					nav_mean_zero_embedding, nav_mean_zero_embedding.transpose())

	updated_nav_covariance = stats.invwishart.rvs(nav_covariance_dof, nav_covariance_scale)

	return updated_nav_mean, updated_nav_covariance


def sample_nav_topic_proportions(args, article_id, old_nav_article_word_assignments):
	num_navs_in_topics = np.zeros(args.num_nav_topics)
	for old_nav_topic_id in old_nav_article_word_assignments[article_id]:
		num_navs_in_topics[old_nav_topic_id] += 1

	return np.random.dirichlet(args.nav_proportions_prior + num_navs_in_topics)


# logsumexp?
def sample_nav_article_word_assignment(args, article_id, word_id, old_nav_topic_means, old_nav_topic_covariances, old_nav_article_proportion, data):
	unnormalized = []
	for nav_topic_id in range(args.num_nav_topics):
		unnormalized.append(old_nav_article_proportion[nav_topic_id] \
			* (1/np.sqrt(np.linalg.det(old_nav_topic_covariances[nav_topic_id])))
			* np.exp((-1/2) * np.matmul())


def sampler(args, last_sample, data):
	# can we normalize out article word assignments?
	old_nav_topic_means, \
	old_nav_topic_covariances, \
	old_nav_article_proportions, \
	old_nav_article_word_assignments = last_sample

	updated_nav_means = []
	updated_nav_covariances = []
	for nav_topic_id in range(args.num_nav_topics):
		updated_nav_mean, updated_nav_covariance = sample_nav_topic_mean_and_variance(
			args, nav_topic_id, old_nav_topic_covariances[nav_topic_id], old_nav_article_word_assignments, data)

		updated_nav_means.append(updated_nav_mean)
		updated_nav_covariances.append(updated_nav_covariance)


if __name__ == '__main__':
	articles, noun_and_verb_vocabulary, noun_and_verb_embeddings, article_noun_and_verb_counts = \
		preprocess.preprocess_articles(langs, datetime(1996, 9, 1), datetime(1997, 1, 1))
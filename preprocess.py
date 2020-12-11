import os
import glob
import torch
import queue
import pickle
import warnings
import xmltodict
import threading
import numpy as np
import pandas as pd
from lazy_load import lazy
from datetime import datetime
import matplotlib.pyplot as plt
from sortedcontainers import SortedSet
from collections import Counter, defaultdict

import spacy
import stanza
from spacy_stanza import StanzaLanguage

from transformers import MarianMTModel, MarianTokenizer

from sklearn.decomposition import PCA

REUTERS_DIRECTORY = 'data/reuters'
LANGUAGE_DIRECTORIES = {
	'en': os.path.join(REUTERS_DIRECTORY, "rcv1"),
	'es': os.path.join(REUTERS_DIRECTORY, "RCV2_Multilingual_Corpus/spanish-latam"),
	'ru': os.path.join(REUTERS_DIRECTORY, "RCV2_Multilingual_Corpus/russian")
}
LANGUAGE_MODELS = {
	'en': lazy(lambda: spacy.load("en_core_web_lg")),
	'es': lazy(lambda: spacy.load("es_core_news_lg")),
	'ru': lazy(lambda: StanzaLanguage(stanza.Pipeline(lang="ru")))
}

def get_mt_tokenizer_and_model(model_name, device):
    return MarianTokenizer.from_pretrained(model_name), MarianMTModel.from_pretrained(model_name).to(device)


def load_stop_words(lang):
	stop_words = set()
	with open('data/stopwords/%s.txt' % lang, 'r') as f:
		for word in f.readlines():
			stop_words.add(word.strip())
	return stop_words

STOP_WORDS = {
	'en': lazy(lambda: load_stop_words("en")),
	'es': lazy(lambda: load_stop_words("es")),
	'ru': lazy(lambda: load_stop_words("ru"))
}

def load_embeddings(lang):
	embeddings = {}
	with open('data/muse/wiki.multi.%s.vec' % lang, 'r') as f:
		for line in f.readlines():
			word, *embedding = line.strip().split(' ')
			embeddings[word.strip().lower()] = np.asarray(embedding).astype(float)
	return embeddings


EMBEDDINGS = {
	'en': lazy(lambda: load_embeddings("en")),
	'es': lazy(lambda: load_embeddings("es")),
	'ru': lazy(lambda: load_embeddings("ru"))
}

NUM_WORKERS = 8

class Article:
	def __init__(self, lang, filepath, date, topics, nouns_and_verbs, named_entities):
		self.lang = lang
		self.filepath = filepath
		self.date = date
		self.topics = topics
		self.nouns_and_verbs = nouns_and_verbs
		self.named_entities = named_entities

	def __str__(self):
		return "%s, %s, %s, %s, %s, %s" % (
			self.lang, self.filepath, self.date, 
			self.topics, self.nouns_and_verbs, self.named_entities
		)


def xdict(maybe_dict):
	if isinstance(maybe_dict, dict):
		return maybe_dict
	return {"key": maybe_dict}


def xlist(maybe_list):
	if isinstance(maybe_list, list):
		return maybe_list
	return [maybe_list]


def get_lang(article):
	parsed['newsitem']['@xml:lang']


def get_date(article):
	return datetime.strptime(article['newsitem']['@date'], "%Y-%m-%d")


def get_title(article):
	return article['newsitem']['title']


def get_headline(article):
	return article['newsitem']['headline']


def get_topics(article):
	topic_codes = set()
	for codes in xlist(article['newsitem']['metadata']['codes']):
		if 'topics' in codes['@class']:
			if 'code' not in codes:
				continue

			for topic_code in xlist(codes['code']):
				topic_codes.add(topic_code['@code'])
	return list(sorted(topic_codes))


def get_body(article):
	text = []
	for key in xdict(article['newsitem']['text']):
		for section in xlist(article['newsitem']['text'][key]):
			text.append(section)
	return ' '.join(text)


def get_articles(langs, date_start=None, date_end=None):
	for lang in langs:
		for xml_filepath in glob.glob(os.path.join(LANGUAGE_DIRECTORIES[lang], "*/*.xml")):
			if lang == 'en' and (date_start or date_end):
				date = datetime.strptime(xml_filepath.split('/')[-2], "%Y%m%d")

				if date_start and date_start > date:
					continue

				if date_end and date_end < date:
					continue

			with open(xml_filepath, 'r') as f:
				try:
					xml_file_contents = xmltodict.parse(f.read())

					if lang != 'en' and (date_start or date_end):
						date = get_date(xml_file_contents)

						if date_start and date_start > date:
							continue

						if date_end and date_end < date:
							continue

					topics = get_topics(xml_file_contents)
					text = get_body(xml_file_contents)
				except:
					print("Unable to parse: %s" % xml_filepath)

				yield lang, xml_filepath, date, topics, text


def explore_articles(langs, date_start=None, date_end=None):
	topics = set()
	lang_counts = Counter()
	topic_counts = defaultdict(Counter)
	for lang, _, xml_file_contents in get_articles(langs, date_start, date_end):
		for topic in get_topics(xml_file_contents):
			topics.add(topic)
			lang_counts[lang] += 1
			topic_counts[lang][topic] += 1

	print(lang_counts)

	df = pd.DataFrame(
		{lang: [counts[topic]/lang_counts[lang] for topic in sorted(topics)] for lang, counts in topic_counts.items()},
		index=list(sorted(topics))
	)
	df.plot.bar(rot=0)
	plt.show()


NOUN_AND_VERB_POS = {'NOUN', 'VERB'}

def get_nouns_and_verbs(parsed):
	nouns_and_verbs = Counter()
	for token in parsed:
		# make sure it's not a named entity!
		if token.ent_iob in [1, 3]:
			continue
		else:
			assert token.ent_iob in [0, 2]

		if token.pos_ not in NOUN_AND_VERB_POS:
			continue

		nouns_and_verbs[token.lemma_.strip().lower()] += 1
	return nouns_and_verbs


NAMED_ENTITY_LABELS = {'PERSON', 'GPE', 'ORG'}

def get_named_entities(lang, parsed):
	named_entities = Counter()
	for named_entity in parsed.ents:
		if named_entity.label_ not in NAMED_ENTITY_LABELS:
			continue

		named_entities[named_entity.text.strip().lower()] += 1
	return named_entities

DONE_PROCESSING = False

def preprocess_articles_worker(q, processed):
	global DONE_PROCESSING

	while not DONE_PROCESSING:
		article_num, (lang, filepath, date, topics, text) = q.get()

		try:
			parsed = LANGUAGE_MODELS[lang](text)

			nouns_and_verbs = get_nouns_and_verbs(parsed)
			named_entities = get_named_entities(lang, parsed)

			article = Article(lang, filepath, date, topics, nouns_and_verbs, named_entities)
			processed.append(article)

			if article_num % 1000 == 0:
				print("%s: Processed %s (%s)" % (datetime.now(), article_num, lang))
				print(str(article))
		except:
			print("Unable to extract: %s" % filepath)

		q.task_done()


BATCH_SIZE = 32

def translate_named_entities(lang, named_entities):
	device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if lang == 'es':
		tokenizer, model = get_mt_tokenizer_and_model('Helsinki-NLP/opus-mt-es-en', device)
	elif lang == 'ru':
		tokenizer, model = get_mt_tokenizer_and_model('Helsinki-NLP/opus-mt-ru-en', device)
	else:
		raise Exception("Unsupported language: %s" % lang)

	translated = {}
	for i in range(0, len(named_entities) / BATCH_SIZE):
		batch = named_entities[i*BATCH_SIZE, (i+1)*BATCH_SIZE]
		inputs = tokenizer.prepare_seq2seq_batch(batch, return_tensors="pt").to(device)
		outputs = model.generate(inputs)

		for (named_entity, output) in zip(batch, outputs):
			translate[named_entity] = tokenizer.decode(output, skip_special_tokens=True).strip().lower()

	return translated


def preprocess_articles(langs, date_start=None, date_end=None, pca_dim=300):
	global DONE_PROCESSING

	extracted_path = os.path.join(REUTERS_DIRECTORY, 'preprocessed/%s--%s.pkl' % (
		date_start.strftime("%m%d%Y"), date_end.strftime("%m%d%Y")))
	if os.path.exists(extracted_path):
		with open(extracted_path, 'rb') as f:
			articles = pickle.load(f)
	else:
		warnings.filterwarnings("ignore", 
			message="Due to multiword token expansion or an alignment issue, the original text has been replaced by space-separated expanded tokens.")
		warnings.filterwarnings("ignore", 
			message="Can't set named entities because of multi-word token expansion or because the character offsets don't map to valid tokens produced by the Stanza tokenizer:")

		articles = []
		q = queue.Queue()

		for thread_num in range(NUM_WORKERS):
			threading.Thread(target=preprocess_articles_worker, args=(q, articles)).start()
		
		for article_num, article in enumerate(get_articles(langs, date_start, date_end)):
			q.put((article_num, article))

		q.join()
		DONE_PROCESSING = True

		with open(extracted_path, 'wb') as f:
			pickle.dump(articles, f)

		print("Processed %s articles!" % len(articles))

	noun_and_verb_vocabulary_path = os.path.join(REUTERS_DIRECTORY, 'preprocessed/%s--%s-nouns-and-verbs.pkl' % (
		date_start.strftime("%m%d%Y"), date_end.strftime("%m%d%Y")))
	noun_and_verb_embeddings_path = os.path.join(REUTERS_DIRECTORY, 'preprocessed/%s--%s-noun-and-verb-embeddings.npy' % (
		date_start.strftime("%m%d%Y"), date_end.strftime("%m%d%Y")))
	noun_and_verb_data_path = os.path.join(REUTERS_DIRECTORY, 'preprocessed/%s--%s-noun-and-verb-data.npy' % (
		date_start.strftime("%m%d%Y"), date_end.strftime("%m%d%Y")))

	if not all(map(os.path.exists, [noun_and_verb_vocabulary_path, noun_and_verb_embeddings_path, noun_and_verb_data_path])):
		lang_counts = Counter(map(lambda article: article.lang, articles))
			
		print("%s articles!" % (len(articles)))
		print(lang_counts)

		###
		# NOUNS AND VERBS
		###

		# now, for our nouns and verbs, we define a vocabulary.
		# keep all nouns and verbs that:
		# 	1) are in less than 30% of documents (ie, contain informative signal)
		#	2) aren't in our (rather aggressive) list of stopwords for each language (ie, contain information signal)
		#	3) are in the set of word embeddings published by MUSE

		noun_and_verb_counts = defaultdict(Counter)
		for article in articles:
			for noun_or_verb in article.nouns_and_verbs:
				if noun_or_verb != noun_or_verb.strip():
					print(noun_or_verb)
				noun_and_verb_counts[article.lang][noun_or_verb] += 1

		print("Unfiltered noun/verb vocab size")
		print({lang: len(noun_and_verb_counts[lang]) for lang in noun_and_verb_counts.keys()})

		noun_and_verbs_by_lang = defaultdict(set)
		noun_and_verb_vocabulary = SortedSet()
		for lang in noun_and_verb_counts:
			for noun_or_verb, count in noun_and_verb_counts[lang].items():
				if count > 0.3 * lang_counts[lang]:
					continue

				if noun_or_verb in STOP_WORDS[lang]:
					continue

				if noun_or_verb not in EMBEDDINGS[lang]:
					continue

				noun_and_verb_vocabulary.add((lang, noun_or_verb))
				noun_and_verbs_by_lang[lang].add(noun_or_verb)

			print("Filtered noun/verb vocab size for %s=%s" % (lang, len(noun_and_verbs_by_lang[lang])))

		noun_and_verb_embeddings = np.array(
			[EMBEDDINGS[lang][noun_or_verb] for lang, noun_or_verb in noun_and_verb_vocabulary]
		)

		# optionally reduce dimensionality (don't need to hold onto PCA matrix)
		if pca_dim < noun_and_verb_embeddings.shape[1]:
			print("Reducing embedding dimensionality from %s to %s" % (noun_and_verb_embeddings.shape[1], pca_dim))
			noun_and_verb_embeddings = PCA(pca_dim).fit_transform(noun_and_verb_embeddings)

		article_noun_and_verb_counts = np.zeros((len(articles), len(noun_and_verb_vocabulary)))
		for article_id, article in enumerate(articles):
			for noun_or_verb, count in article.nouns_and_verbs.items():
				if (article.lang, noun_or_verb) not in noun_and_verb_vocabulary:
					continue

				article_noun_and_verb_counts[article_id][noun_and_verb_vocabulary.index((article.lang, noun_or_verb))] = count 

		with open(noun_and_verb_vocabulary_path, 'wb') as f:
			pickle.dump(noun_and_verb_vocabulary, f)

		np.save(noun_and_verb_embeddings_path.strip(".npy"), noun_and_verb_embeddings)

		np.save(noun_and_verb_data_path.strip(".npy"), article_noun_and_verb_counts)

		print("Wrote nouns and verbs of size (%s, %s) to %s" % (*article_noun_and_verb_counts.shape, noun_and_verb_data_path))
	else:
		with open(noun_and_verb_vocabulary_path, 'rb') as f:
			noun_and_verb_vocabulary = pickle.load(f)

		noun_and_verb_embeddings = np.load(noun_and_verb_embeddings_path)

		article_noun_and_verb_counts = np.load(noun_and_verb_data_path)

	###
	# NAMED ENTITIES 
	###

	named_entity_vocabulary_path = os.path.join(REUTERS_DIRECTORY, 'preprocessed/%s--%s-named-entities.pkl' % (
		date_start.strftime("%m%d%Y"), date_end.strftime("%m%d%Y")))
	named_entity_data_path = os.path.join(REUTERS_DIRECTORY, 'preprocessed/%s--%s-named-entities-data.npy' % (
		date_start.strftime("%m%d%Y"), date_end.strftime("%m%d%Y")))

	if not all(map(os.path.exists, [named_entity_vocabulary_path, named_entity_data_path])):
		named_entities = set()
		for article in articles:
			for named_entity in article.named_entities:
				named_entities.add((article.lang, named_entity))
		named_entities = list(sorted(named_entities))

		print("Ungrouped named entities: %s" % len(named_entities))

		es_named_entities = [named_entity for (lang, named_entity) in named_entities if lang == 'es']
		es_named_entities_translated = translate_named_entities('es', es_named_entities)
		ru_named_entities = [named_entity for (lang, named_entity) in named_entities if lang == 'ru']
		ru_named_entities_translated = translate_named_entities('ru', ru_named_entities)

		grouped_named_entity_counts = Counter()
		for article in articles:
			for named_entity in article.named_entities:
				if article.lang == 'es':
					named_entity = es_named_entities_translated[named_entity]
				elif article.lang == 'ru':
					named_entity = ru_named_entities_translated[named_entity]			

				grouped_named_entity_counts[named_entity] += 1

		named_entity_vocabulary = SortedSet()
		for named_entity, count in grouped_named_entity_counts.items():
			if count > 5 and count < 0.8 * len(articles):
				named_entity_vocabulary.add(named_entity)

		print("Grouped named entities: %s" % len(named_entity_vocabulary))

		article_named_entity_counts = np.zeros((len(articles), len(named_entity_vocabulary)))
		for article_id, article in enumerate(articles):
			for named_entity, count in article.named_entities.items():
				if article.lang == 'es':
					named_entity = es_named_entities_translated[named_entity]
				elif article.lang == 'ru':
					named_entity = ru_named_entities_translated[named_entity]				

				if named_entity not in named_entity_vocabulary:
					continue

				article_named_entity_counts[article_id][named_entity_vocabulary.index(named_entity)] += count 

		with open(named_entity_vocabulary_path, 'wb') as f:
			pickle.dump(named_entity_vocabulary, f)

		np.save(named_entity_data_path.strip(".npy"), article_named_entity_counts)
	else:
		with open(named_entity_vocabulary_path, 'rb') as f:
			named_entity_vocabulary = pickle.load(f)

		article_named_entity_counts = np.load(named_entity_data_path)

	return articles, \
		noun_and_verb_vocabulary, \
		torch.tensor(noun_and_verb_embeddings), \
		torch.tensor(article_noun_and_verb_counts).float(), \
		named_entity_vocabulary, \
		torch.tensor(article_named_entity_counts).float()

if __name__ == '__main__':
	preprocess_articles(['en', 'es', 'ru'], datetime(1996, 9, 1), datetime(1996, 9, 2))

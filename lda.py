import re
import warnings
import pandas as pd
import numpy as np
import time

import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

import pyLDAvis.gensim
import matplotlib.pyplot as plt

# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

warnings.filterwarnings("ignore", category=DeprecationWarning)

stop_words = stopwords.words('english')

stop_words.extend(
    ['from', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get',
     'do', 'done', 'try', 'many', 'some', 'rather', 'lot', 'make', 'want', 'seem', 'run',
     'need', 'even', 'also', 'may', 'take', 'come', 'mr', 'percent', 'dr', 'ms', 'monday',
     'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])


def read_data():
    data = pd.read_json("./data/data.json")
    texts = data["text"].drop_duplicates()
    print(texts)
    return texts.values


def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
        yield(sent)


def process_words(data_words):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stopwords, Form Bigrams and Trigrams
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_words]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

    return texts


def preprocess():
    docs = read_data()
    data_words = list(sent_to_words(docs))
    preprocessed_data = process_words(data_words)

    return preprocessed_data


def create_dict_and_corpus(texts):

    # create dict file
    DICT_FILE = "./artifacts/news.topics.dict"
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=200000)
    # dictionary.filter_n_most_frequent(5)
    dictionary.save(DICT_FILE)

    print("dictionary tokenid", len(dictionary.token2id))
    print("words", texts)

    corpus = list(map(lambda x: dictionary.doc2bow(x), texts))
    print("corpus length", corpus)

    return dictionary, corpus


def train(texts, num_topics=10):
    dictionary, corpus = create_dict_and_corpus(texts)

    NUM_TOPICS = num_topics
    ldamodel_file = "./artifacts/news.topics.model"
    ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                               num_topics=NUM_TOPICS,
                                               id2word=dictionary,
                                               passes=15)
    ldamodel.save(ldamodel_file)

    topics = ldamodel.print_topics(num_words=20)
    for topic in topics:
        print(topic)

    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary=ldamodel.id2word)
    pyLDAvis.save_html(vis, 'lda.html')


def get_topics():
    ldamodel = gensim.models.ldamodel.LdaModel.load("./artifacts/news.topics.model")
    topics = ldamodel.print_topics(num_words=20)
    for topic in topics:
        print(topic)


def score():
    dictionary = gensim.corpora.dictionary.Dictionary.load("./artifacts/news.topics.dict")
    ldamodel = gensim.models.ldamodel.LdaModel.load("./artifacts/news.topics.model")

    data = pd.read_json("./data/data.json")
    data = data.sort_values(by=['doc_id', "text_id"])

    num_days = data["doc_id"].unique()
    dates = data["date"].unique()
    print(num_days, dates)

    result_array = []

    for _doc_id, _date in zip(num_days, dates):

        filtered_data = data.loc[data['doc_id'] == _doc_id]
        current_day_text = "\n".join(filtered_data["text"].values)

        data_words = list(sent_to_words([current_day_text]))
        processed_new_doc = process_words(data_words)[0]

        topics = ldamodel.get_document_topics(
            dictionary.doc2bow(processed_new_doc),
            minimum_probability=0.2
        )

        main_topics = [_topic for _topic in topics
                       if _topic[1] == max(_topic[1] for _topic in topics)]

        topic_name = ldamodel.show_topic(main_topics[0][0])
        probability = main_topics[0][1]

        for _topic_name in topic_name:
            _topic, topic_prob = _topic_name
            result_array.append(
                {
                    "date": np.datetime_as_string(_date, unit='D'),
                    "topic": _topic,
                    "probability": topic_prob
                }
            )

    print(result_array)

    vis_df = pd.DataFrame(result_array, columns=["date", "topic", "probability"])
    vis_df = vis_df.sort_values(by="date")
    print(vis_df)

    plt.plot(vis_df["date"], vis_df["probability"], "bo")
    plt.savefig("analysis.jpg")


        # for _topic in topics:
        #     print("Topics", _topic)
        #
        #     topic_name = ldamodel.show_topic(_topic[0])
        #     probability = _topic[1]
        #
        #     print([_t[0] for _t in topic_name], [_t[1] for _t in topic_name])


# processed_texts = preprocess()
# train(processed_texts)

# get_topics()

score()




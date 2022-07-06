import nltk
from pprint import pprint
from nltk.corpus import movie_reviews
import random

'''
nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])
'''

# get words from corpus
words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
print(len(words))

# remove stop words
stopwords = nltk.corpus.stopwords.words("english")
words = [w for w in words if w.lower() not in stopwords]
print(len(words))

text = """
For some quick analysis, creating a corpus could be overkill.
If all you need is a word list,
there are simpler ways to achieve that goal."""
pprint(nltk.word_tokenize(text), width=79, compact=True)
text = nltk.word_tokenize(text)
text = [w for w in text if w.isalpha()]

documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)
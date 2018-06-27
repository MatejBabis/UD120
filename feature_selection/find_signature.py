#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


# The words (features) and authors (labels), already largely processed.
# These files should have been created from the previous (Lesson 10)
# mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "r"))
authors = pickle.load(open(authors_file, "r"))


# test_size is the percentage of events assigned to the test set (the
# remainder go into training)
# feature matrices changed to dense representations for compatibility with
# classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

# a classic way to overfit is to use a small number
# of data points and a large number of features;
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

# use a decision tree
from sklearn import tree

classifier = tree.DecisionTreeClassifier()
classifier.fit(features_train, labels_train)

# print accuracy
print "overfitted accuracy:", classifier.score(features_test, labels_test)
# NOTE: this might lead to a slightly different answer than the one
# from the course because of different version of nltk.corpus

# find maximum feature importance and its index
feature_importances = classifier.feature_importances_
# need to use numpy as stock sort() throws a parsing error
import numpy as np
indices = np.argsort(feature_importances)[::-1]

# print at most 10 most important features
for i in range(10):
    if feature_importances[indices[i]] > 0.2:
        print "Important feature no.", indices[i], "=", feature_importances[indices[i]]
# NOTE: same situation here - result might be slightly different

print "The most discriminating word is:", vectorizer.get_feature_names()[indices[0]]

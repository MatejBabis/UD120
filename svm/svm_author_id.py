#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# reduce the dataset to increase speed
# features_train = features_train[:len(features_train) / 100]
# labels_train = labels_train[:len(labels_train) / 100]

# test different C values
# c_values = [1., 10., 100., 1000., 10000.]
c_values = [10000.]

for c in c_values:
    classifier = SVC(C=c, kernel="rbf")
    # fitting timer
    t0 = time()
    classifier.fit(features_train, labels_train)
    print "training time:", round(time() - t0, 3), "s"

    # prediction
    pred = classifier.predict(features_test)

    # prediction for specified cases
    # for i in [10, 26, 50]:
    #     print "prediction " + str(i) + ":", pred[i]

    # count prediction occurence for Chris (1)
    chris_occ = 0
    for p in pred:
        if p == 1:
            chris_occ += 1
    print "Chris was predicted " + str(chris_occ) + " times."

    # print accuracy
    print "accuracy for C = " + str(c) + ":", classifier.score(features_test, labels_test)

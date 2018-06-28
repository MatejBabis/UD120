#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))

# add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

classifier = tree.DecisionTreeClassifier()

classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test)

conf_matrix = confusion_matrix(labels_pred, labels_test)

print conf_matrix

print "Precision =", conf_matrix[1][1] / float(conf_matrix[1][0] + conf_matrix[1][1])
print "Recall =", conf_matrix[1][1] / float(conf_matrix[0][1] + conf_matrix[1][1])

##############
# "made-up" dataset
print "\nMade-up dataset"

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

conf_matrix = confusion_matrix(predictions, true_labels)

print conf_matrix

print "Precision =", conf_matrix[1][1] / float(conf_matrix[1][0] + conf_matrix[1][1])
print "Recall =", conf_matrix[1][1] / float(conf_matrix[0][1] + conf_matrix[1][1])

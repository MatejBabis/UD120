#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# read in data dictionary, convert to numpy array
data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))

# remove the spreasheet mistake
data_dict.pop("TOTAL", 0)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# find the four biggest outliers
outliers = []
for key in data_dict:
    salary = data_dict[key]['salary']
    bonus = data_dict[key]['bonus']

    # ignore invalid values
    if salary == 'NaN' or bonus == 'NaN':
        continue

    # find the biggest "bandits"
    if int(salary) > 1000000 and int(bonus) > 5000000:
        outliers.append((key, int((salary - bonus)**2)))

print "The biggest outliers are:"
for outlier in outliers:
    print outlier

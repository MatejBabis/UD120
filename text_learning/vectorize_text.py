#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append("../tools/")
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

# temp_counter is a way to speed up the development--there are
# thousands of emails from Sara and Chris, so running over all of them
# can take a long time
# temp_counter helps you only look at the first 200 emails in the list so you
# can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        path = os.path.join('..', path[:-1])
        print path
        email = open(path, "r")

        # use parseOutText to extract the text from the opened email
        parsed_email = parseOutText(email)

        temp_string = ""
        for word in parsed_email.split(" "):
            # check for unwanted words; don't include them
            if word not in ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]:
                temp_string += " " + word
        # remove leading / trailing whitespace
        parsed_email = temp_string.strip()

        # append the text to word_data
        word_data.append(parsed_email)

        # append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name == "sara":
            from_data.append(0)
        else:
            from_data.append(1)
        # NOTE: should probably include some error handling here...

        email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump(word_data, open("your_word_data.pkl", "w"))
pickle.dump(from_data, open("your_email_authors.pkl", "w"))

# TfIdf vectorization here
sw = stopwords.words("english")     # stopwords
# vectorizer
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
vectorizer.fit(word_data)
print "Number of unique words in TfIdf:", len(vectorizer.get_feature_names())
# NOTE: this leads to a different answer (38821) than the one
# from the course because of different version of nltk.corpus

print "Word number 34597 is:", vectorizer.get_feature_names()[34597]
# NOTE: same situation here - result is different ("statesenron")

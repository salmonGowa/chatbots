import numpy as np
import tensorflow
from nltk.stem.lancaster import LancasterStemmer
import random
import json
import tflearn


stemmer=LancasterStemmer()
with open('intents.json') as file:
    data=json.load(file)

words=[]
labels=[]
docs_x=[]
docs_y=[]

for intent in data['intent']:
    for pattern in intent['patterns']:
        wrds=nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(pattern)

    if intent['tags'] not in labels:
        labels.append(intent['tags'])

words=[stemmer.stem(w.lower()) for w in words]
words=sorted(list(set(words)))

labels=sorted(labels)
training=[]
output=[]
out_empty=[0 for _ in range(len(classes))]

for x, doc in enumerate(docs_x):
    bag=[]
    wrds=[stemmer.stem(w)for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row=out_empty[:]

    output_row[labels.index(docs_y[x])]=1

    training.append(bag)
    output.append(output_row)

    training=np.array(training)
    output=np.array(output)

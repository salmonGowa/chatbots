import numpy
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
docs=[]

for intent in data['intent']:
    for pattern in intent['patterns']:
        wrds=nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent['tags'] not in labels:
        labels.append(intent['tags'])
        

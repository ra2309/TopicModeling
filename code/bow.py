import pandas as pd
import nltk
import pickle
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import matplotlib
import matplotlib.pyplot as plt

stemmer = LancasterStemmer()
data=pd.read_pickle('../../data/df_tokens.pkl')

ignore_words = stopwords.words('english')
# Bag of words
f = lambda tokens: Counter([word for word in tokens if word not in ignore_words and word.isalpha()])

data['bow'] = (pd.DataFrame(data['tokens'].apply(f).values.tolist())
               .fillna(0)
               .astype(int)
               .values
               .tolist())
data.to_pickle('../../data/bow.pkl')

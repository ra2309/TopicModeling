# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import codecs
from sklearn.model_selection import train_test_split
import pandas as pd

stemmer = LancasterStemmer()

folders = os.listdir('../../data/mini_newsgroups')
X = []
y = []
for f in folders:
	files = os.listdir('../../data/mini_newsgroups/'+f)
	for fil in files:
		with open('../../data/mini_newsgroups/'+f+'/'+fil,'rb') as content:
			X.append(content.read().decode("latin-1").strip())
			y.append(f)

data = pd.DataFrame(columns=['content','class'])

data['content'] = X
data['class'] = y
data.to_pickle('../../data/mini_dataframe.pkl')



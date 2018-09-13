import pandas as pd
import nltk
import pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import os



lib = ['atheism','motor','graphic','windows','hardware','mac','x','sale','auto','baseball','hockey','crypt','electronic','medicine','space','christian','guns','mideast','politics','religion']

folders = os.listdir('../../data/mini_newsgroups')
y = []
i = 0
count=0
for f in folders:
	y.append(i)
	files = os.listdir('../../data/mini_newsgroups/'+f)
	for fil in files:
		X = []
		with open('../../data/mini_newsgroups/'+f+'/'+fil,'rb') as content:
			for l in lib:
				c = content.read().decode("latin-1").strip()
				X.append(c.count(l))
			y_ = X.index(max(X))
			if y_ == y:
				count=count+1
	i=i+1

print(count/i)
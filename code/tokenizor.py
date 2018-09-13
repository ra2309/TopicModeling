import pandas as pd
import nltk
import pickle
import numpy as np
with open('../../data/mini_dataframe.pkl','rb') as f:
	data=pickle.load(f)

# Tokenization
df = pd.DataFrame(data)
df['tokens'] = df.apply(lambda row: nltk.word_tokenize(row['content']), axis=1)

df.to_pickle('../../data/df_tokens.pkl')


from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

data = pd.read_pickle('../../data/df_bow.pkl')

X_train, X_test, y_train, y_test = train_test_split(data['bow'], data['class'], test_size = 0.33, random_state=20)

train = pd.DataFrame()
test = pd.DataFrame()

train['X'] = X_train
train['y'] = y_train
test['X'] = X_test
test['y'] = y_test


with open('../../data/train.pkl', 'wb') as f:
	pickle.dump(train, f)
with open('../../data/test.pkl', 'wb') as f:
	pickle.dump(test, f)
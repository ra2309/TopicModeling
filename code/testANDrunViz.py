import pandas as pd
import pickle
import tensorflow as tf
import tflearn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import NNVisualization

test = pd.read_pickle('../../data/test.pkl')

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(test['y'])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)
test['y'] = pd.DataFrame(y.reshape(-1, len(y)))
# Build neural network
net = tflearn.input_data(shape=[None, len(test['X'])])
net = tflearn.dropout(net, .2)
net = tflearn.dropout(net, .4)
net = tflearn.fully_connected(net, len(test['y']), activation='tanh')
net = tflearn.regression(net)

network = NNVisualization.DrawNN([16,16,8,20 ])
network.draw()
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.load('../../data/model.tflearn-improved2')
preds = model.predict(list(test['X']))
predicted = np.argmax(preds,axis=1)
actual = np.argmax(y,axis=1)
output = actual == predicted
print(sum(output)/len(output))
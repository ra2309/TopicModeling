import pandas as pd
import pickle
import numpy as np
import time
import tensorflow as tf
import tflearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example

# integer encode
label_encoder = LabelEncoder()
# binary encode

train = pd.read_pickle('../../data/train.pkl')

X = np.array(train['X'])
categories = np.array(train['y'])
integer_encoded = label_encoder.fit_transform(categories)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)
start_time = time.time()
# reset underlying graph data

tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(X[0])])
net = tflearn.dropout(net, .2)
net = tflearn.dropout(net, .4)
net = tflearn.fully_connected(net, len(y[0]), activation='tanh')
net = tflearn.regression(net)
 
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(list(X), list(y), n_epoch=100, batch_size=16, show_metric=True)
model.save('../../data/model.tflearn-improved2')

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")

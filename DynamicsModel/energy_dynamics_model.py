import tensorflow as tf
from tensorflow.contrib import rnn
import os
import sys
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from sklearn import pipeline
import itertools
sys.path.append('/Users/stefruinard/Desktop/AI/DynamicsModel/')
from pipeline import Preprocessing

data = pd.read_csv('/Users/stefruinard/Documents/ML6/DataECC/exp3_010.csv')

x_train = data.loc[1000000:2020000, :]
x_test = data.loc[2020000:2030000, :]

print('------Data and Packages Loaded------')

mean_window = 4

#size of batch
batch_size=128

pipeline = Preprocessing(batch_size=batch_size, mean_window=mean_window, skip_n_frames=12,X_train=x_train, X_test=x_test, lag_period=10)
batch_x,batch_y= pipeline._preprocess()

n_time_steps = np.shape(batch_x)[1]/(7+mean_window*3)
#hidden LSTM units
num_units=128
#rows of 28 pixels
n_input=(7+3*mean_window)
n_input
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=7

tf.reset_default_graph()
#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

#define constants
#unrolled through 28 time steps






#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x_placeholder=tf.placeholder(tf.float32,[None,n_time_steps, n_input])
#input label placeholder
y_label=tf.placeholder(tf.float32,[None,n_classes])

input_reshaped = tf.unstack(x_placeholder, n_time_steps, 1)

#defining the network
lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input_reshaped,dtype=tf.float32)

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
# prediction=tf.matmul(outputs[-1],out_weights)+out_bias
prediction=tf.layers.dense(outputs[-1], n_classes)
#loss_function
loss=tf.losses.mean_squared_error(predictions=prediction, labels=y_label)
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    loss_list = []
    iter=1
    while iter<2000:
        batch_x,batch_y= pipeline._preprocess()
        batch_x = np.reshape(np.array(batch_x), newshape=[batch_size,int(n_time_steps),n_input])

        sess.run(opt, feed_dict={x_placeholder:batch_x, y_label: batch_y})

        if iter %10==0:
            los=sess.run(loss,feed_dict={x_placeholder:batch_x, y_label:batch_y})
            print("For iter ",iter)
            print("Loss ",los)
            print("__________________")
        loss_list.append(los)
        iter=iter+1


import matplotlib.pyplot as plt
plt.style.use('ggplot')
x_axis = np.arange(np.shape(loss_list)[0])
plt.plot(x_axis, loss_list)
plt.show()





tf.reset_default_graph()


x_placeholder=tf.placeholder(tf.float32,[None,int(n_time_steps* n_input)])
#input label placeholder
y_label=tf.placeholder(tf.float32,[None,n_classes])


layer_1 = tf.layers.dense(units=500, inputs=x_placeholder, activation=tf.nn.relu)
layer_2 = tf.layers.dense(units=250, inputs=layer_1, activation=tf.nn.relu)
layer_3 = tf.layers.dense(units=50, inputs=layer_2, activation=tf.nn.relu)
prediction = tf.layers.dense(inputs=layer_3, units=n_classes)
loss=tf.losses.mean_squared_error(predictions=prediction, labels=y_label)
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list_2 = []
    iter = 1
    while iter < 2000:
        batch_x, batch_y = pipeline._preprocess()

        sess.run(opt, feed_dict={x_placeholder: batch_x, y_label: batch_y})

        if iter % 10 == 0:
            los = sess.run(loss, feed_dict={x_placeholder: batch_x, y_label: batch_y})
            print("For iter ", iter)
            print("Loss ", los)
            print("__________________")
        loss_list_2.append(los)
        iter = iter + 1

import matplotlib.pyplot as plt
plt.style.use('ggplot')
x_axis = np.arange(np.shape(loss_list)[0])
plt.plot(x_axis, loss_list)
plt.plot(x_axis, loss_list_2)
plt.show()
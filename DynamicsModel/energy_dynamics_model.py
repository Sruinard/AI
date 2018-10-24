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
import matplotlib.pyplot as plt
sys.path.append('/Users/stefruinard/Desktop/AI/DynamicsModel/')
from pipeline import Preprocessing
import matplotlib.pyplot as plt
from deep_networks import Networks

all_files = ['/Users/stefruinard/Documents/ML6/DataECC/exp3_006.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_007.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_008.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_009.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_010.csv']
df = pd.concat((pd.read_csv(f) for f in all_files))
x_train = df[:12000000]
x_test = df[12000000:]
print('------Data and Packages Loaded------')


base_line_validation = []
loss_sequences = []
validation_sequences = []
batch_size=128
temp_batch_size = 1280


skip_n_frames_list = [1,4,8,10,16,20,3,6,9,15,2,4,8,10,16,20,3,6,9,15]
mean_window_list = [2,3,9,20,2,11,8,4,2,10,2,3,9,20,2,11,8,4,2,10]
lag_period_list = [20,8,10,2,12,4,3,9,13,2,20,8,10,1,12,4,3,9,13,2]
lstm_units_list = [128,128,128,64,64,64,12,12,12,24,24,24,50,50,50,100,100,100,200,200]
type_list = ["LSTM"]*10 + ['dense']*10
n_units_list = [[500,250,10], [100,100,100], [20,10,10], [50,25,10],[50,50,50], [60,40,20],[100,50,10], [150,100,50], [75,50,25], [30,20,10]]*2

for i in range(1):

    # initialize variables
    mean_window, skip_n_frames, lag_period, type, n_units, lstm_units = skip_n_frames_list[i], mean_window_list[i], lag_period_list[i], type_list[i], n_units_list[i], lstm_units_list[i]
    tf.reset_default_graph()
    Preprocess_input = Preprocessing(batch_size=batch_size, mean_window=mean_window, skip_n_frames=skip_n_frames,X_train=x_train, X_test=x_test, lag_period=lag_period)
    Network = Networks(type=type, n_special_layers=0, n_dense_layers=3, n_units=n_units, lstm_n_units=lstm_units, n_classes=7, Preprocess_input=Preprocess_input, learning_rate=0.001)
    init = tf.global_variables_initializer()
    np.random.seed(5)
    with tf.Session() as sess:
        sess.run(init)
        loss_list = []
        validation_list = []
        iter = 1
        while iter < 500:

            batch_x, batch_y = Preprocess_input._preprocess()
            if Network.type =='LSTM':
                batch_x = np.reshape(np.array(batch_x), newshape=[batch_size, int(Network.n_time_steps), Network.n_input])

            sess.run(Network.opt, feed_dict={Network.x_placeholder: batch_x, Network.y_label: batch_y})

            if iter % 100 == 0:
                los = sess.run(Network.loss, feed_dict={Network.x_placeholder: batch_x, Network.y_label: batch_y})
                print("For iter ", iter)
                print("Loss ", los)
                print("__________________")
                loss_list.append(los)

            if iter%100 == 0:

                validation_batch_x, validation_batch_y = Preprocess_input._preprocess(validation_flag=True, temp_batch_size=temp_batch_size)
                if Network.type == 'LSTM':
                    validation_batch_x = np.reshape(np.array(validation_batch_x), newshape=[temp_batch_size, int(Network.n_time_steps), Network.n_input])
                validation = sess.run(Network.loss, feed_dict={Network.x_placeholder: validation_batch_x, Network.y_label: validation_batch_y})
                validation_list.append(validation)
                if i == 19:
                    input_as_prediction = np.sum(np.mean(np.square((validation_batch_x.iloc[:, :7] - validation_batch_y))))
                    base_line_validation.append(input_as_prediction)
            if iter ==499:
                loss_sequences.append((loss_list,mean_window, skip_n_frames, lag_period, type, n_units, lstm_units))
                validation_sequences.append(validation_list)
            iter = iter + 1



















i=2
base_line_validation = []
loss_sequences = []
validation_sequences = []
batch_size=128


skip_n_frames_list = [1,4,8,10,16,20,3,6,9,15,2,4,8,10,16,20,3,6,9,15]
mean_window_list = [2,3,9,20,2,11,8,4,2,10,2,3,9,20,2,11,8,4,2,10]
lag_period_list = [20,8,10,2,12,4,3,9,13,2,20,8,10,1,12,4,3,9,13,2]
lstm_units_list = [128,128,128,64,64,64,12,12,12,24,24,24,50,50,50,100,100,100,200,200]
type_list = ["LSTM"]*10 + ['dense']*10
n_units_list = [[500,250,10], [100,100,100], [20,10,10], [50,25,10],[50,50,50], [60,40,20],[100,50,10], [150,100,50], [75,50,25], [30,20,10]]*2

mean_window, skip_n_frames, lag_period, type, n_units, lstm_units = skip_n_frames_list[i], mean_window_list[i], lag_period_list[i], 'LSTM' , n_units_list[i], lstm_units_list[i] #type_list[i]
tf.reset_default_graph()
Preprocess_input = Preprocessing(batch_size=batch_size, mean_window=mean_window, skip_n_frames=skip_n_frames,X_train=x_train, X_test=x_test, lag_period=lag_period)
Network = Networks(type=type, n_special_layers=0, n_dense_layers=3, n_units=n_units, lstm_n_units=lstm_units, n_classes=7, Preprocess_input=Preprocess_input, learning_rate=0.001)
init = tf.global_variables_initializer()
np.random.seed(5)
with tf.Session() as sess:
    sess.run(init)
    loss_list = []
    validation_list = []
    iter = 1
    while iter < 5000:
        if iter <1400:
            np.random.randint(20)
            iter += 1
            continue
        if iter % 100:
            print('still running')
            temp_batch_size = 1280
            validation_batch_x, validation_batch_y = Preprocess_input._preprocess(validation_flag=True,
                                                                                  temp_batch_size=temp_batch_size)
            if Network.type == 'LSTM':
                validation_batch_x = np.reshape(np.array(validation_batch_x),
                                                newshape=[temp_batch_size, int(Network.n_time_steps), Network.n_input])
            validation = sess.run(Network.loss, feed_dict={Network.x_placeholder: validation_batch_x,
                                                           Network.y_label: validation_batch_y})
            validation_list.append(validation)

        # if i == 2:
        #     error = validation_batch_x.iloc[:, :7] - validation_batch_y
        #     input_as_prediction = np.sum(np.mean(np.square((validation_batch_x.iloc[:, :7] - validation_batch_y))))
        #     base_line_validation.append(input_as_prediction)
        iter += 1




for i in range(2):
    np.random.seed(5)
    for i in range(3):
        print(np.random.randint(20))



for i in range(2):
    np.random.seed(5)
    for i in range(3):
        if i<2:
            np.random.randint(20)
            continue
        print(np.random.randint(20))







plt.style.use('ggplot')

val_mean_list_numbers = []
train_mean_list_numbers = []
val_means = []
train_means = []
for i in range(20):
    val_mean_list_numbers.append(validation_sequences[i][-100:])
    train_mean_list_numbers.append(loss_sequences[i][0][-100:])

for i in range(20):
    val_means.append(np.mean(val_mean_list_numbers[i]))
    train_means.append(np.mean(train_mean_list_numbers[i]))




i = np.argmin(val_means)
y_loss_data = loss_sequences[i][0]
labels = str(loss_sequences[i][1:7])
x_axis = np.arange(np.shape(y_loss_data)[0])
plot = plt.plot(x_axis, y_loss_data, label=('test: '+ labels))
y_loss_val = validation_sequences[i]
plot = plt.plot(x_axis, y_loss_val, label=('validation: '+labels))
plt.legend()
plt.show()

for i in range(20):
    y_loss_data = loss_sequences[i][0]
    labels = str(loss_sequences[i][1:7])
    x_axis = np.arange(np.shape(y_loss_data)[0])
    plot = plt.plot(x_axis, y_loss_data)
    y_loss_val = validation_sequences[i]
    plot = plt.plot(x_axis, y_loss_val)
    plt.legend()
    plt.show()





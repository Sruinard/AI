#Note several lines are hided from sight
#Hidden lines contain imported packages and network settings
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import tqdm
import sys
sys.path.append('/Users/stefruinard/Desktop/AI/DynamicsModel')
from pipeline import DataSelection, Preprocessing, SpecifyBatching
from deep_networks import AttrDict, Networks
from drift_preprocessing import DriftBatch

all_files = ['/Users/stefruinard/Documents/ML6/DataECC/exp3_013.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_010.csv'] #'/Users/stefruinard/Documents/ML6/DataECC/exp3_006.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_007.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_008.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_009.csv',
df = pd.concat((pd.read_csv(f) for f in all_files))

relevant_columns = ['V_source', 'switch_U', 'switch_V', 'switch_W', 'I_U', 'I_V','I_W', 'sensor_torque', 'encoder_rpm', 'temperature_board']
action_cols = ['switch_U', 'switch_V', 'switch_W']
cols_to_standardize = ['V_source', 'I_U', 'I_V', 'I_W', 'sensor_torque', 'encoder_rpm', 'temperature_board']

n_models = 20
sequence_n_dense_layers = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
sequence_n_lstm_layers = [2,1,1,1,2,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
sequence_dense_units = [[],[],[],[], [10],[20],[30],[40],[20,10],[30,10],[50,20],[100,40],[100,60,20],[80,40,20],[10,10,10],[20,10,10],[100,80,60,40],[100,100,50,20],[60,60,40,20],[20,20,10,10],]
sequence_lstm_units = [[128,64],[128],[128],[64],[128,64],[64],[64],[64],[128],[128],[],[],[],[],[],[],[],[],[],[]]
sequence_lag = [0,1,0,1,0,1,2,1,2,1,3,2,3,4,5,6,4,3,2,1]
sequence_mean_window = [2,4,6,8,10,2,4,6,8,10,2,4,6,8,10,2,4,6,8,10]
sequence_skip_frames = [2,4,2,4,5,10,5,10,20,5,4,2,4,2,4,1,5,10,4,2]
sequence_network_type = ['LSTM']*10 + ['dense']*10
network_specifications = list(zip(sequence_n_dense_layers, sequence_n_lstm_layers, sequence_dense_units, sequence_lstm_units, sequence_lag, sequence_mean_window,sequence_skip_frames, sequence_network_type))

Database = DataSelection(df, 0.8, relevant_columns=relevant_columns)
PreProcesser = Preprocessing(data_for_fitting=Database.train_df)
training_data = PreProcesser._transform(data_to_transform=Database.train_df)
validation_data = PreProcesser._transform(data_to_transform=Database.val_df)

print('------ Packages Loaded, Data Selected and Transformed ------')




for model in range(20):

    tqdm.tqdm.write("Start of model number:" + str(model) + ' out of ' + str(n_models))
    # specify tensorboard location
    tf.reset_default_graph()
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = '/Users/stefruinard/Desktop/Graphs/'
    logdir = "{}/run-{}-{}/".format(root_logdir, now, 'model' + str(model))

    # specify_network
    n_dense_layers, n_lstm_layers, units_layer_n_dense, units_layer_n_lstm, lag, mean_window, skip_n_frames, network_type = \
    network_specifications[model]
    my_network = Networks(n_dense_layers=n_dense_layers, n_lstm_layers=n_lstm_layers,
                          units_layer_n_dense=units_layer_n_dense, units_layer_n_lstm=units_layer_n_lstm, logdir=logdir,
                          network_type=network_type, lag=lag, mean_window=mean_window)
    summarizer, file_writer = my_network.__metrics__()
    minimizer = my_network.__optimize__()

    # specify batch settings in line with network
    batch_size_train = 32
    batch_size_validation = 128
    BatchSpecsTraining = SpecifyBatching(batch_size=batch_size_train, df=training_data,
                                         neural_network_type=network_type, skip_n_frames=skip_n_frames,
                                         mean_window=mean_window, lag=lag)
    BatchSpecsValidation = SpecifyBatching(batch_size=batch_size_validation, df=validation_data,
                                           neural_network_type=network_type, skip_n_frames=skip_n_frames,
                                           mean_window=mean_window, lag=lag)
    optimal_loss = 100
    optimal_loss_drift = 100
    n_epochs = 10000
    store_every_n_iterations = 50
    # train the model and write to tensorboard
    start_optimizing_for_drift = 8500
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.seed(5)
        for epoch in tqdm.tqdm(range(n_epochs)):
            batch_x, batch_y = BatchSpecsTraining._select_batch()
            feed_dict = my_network.__create_feed_dict__(batch_x, batch_y)

            if epoch > start_optimizing_for_drift:
                DriftBatchCharacteristics = DriftBatch(batch_x=batch_x, batch_y=batch_y, lag=lag,
                                                       mean_window=mean_window, batch_size=batch_size_train,
                                                       network_type=network_type, decorrelate_flag=True,
                                                       network=my_network)
                DriftBatchCharacteristics.create_batch(sess, batch_size_train)
                batch_x, batch_y = DriftBatchCharacteristics.batch_x, DriftBatchCharacteristics.batch_y
                feed_dict = my_network.__create_feed_dict__(batch_x, batch_y)

            sess.run(minimizer, feed_dict=feed_dict)

            if epoch % store_every_n_iterations == 0:
                batch_x, batch_y = BatchSpecsValidation._select_batch()
                feed_dict = my_network.__create_feed_dict__(batch_x, batch_y)
                # DRIFTTTTTT
                if epoch > start_optimizing_for_drift:
                    DriftBatchCharacteristics = DriftBatch(batch_x=batch_x, batch_y=batch_y, lag=lag,
                                                           mean_window=mean_window, batch_size=batch_size_validation,
                                                           network_type=network_type, decorrelate_flag=True,
                                                           network=my_network)
                    DriftBatchCharacteristics.create_batch(sess, batch_size_validation)
                    batch_x, batch_y = DriftBatchCharacteristics.batch_x, DriftBatchCharacteristics.batch_y
                    feed_dict = my_network.__create_feed_dict__(batch_x, batch_y)
                    summary_str = summarizer.eval(feed_dict=feed_dict)
                    file_writer.add_summary(summary_str, epoch)
                    temp_optimal_loss = sess.run(my_network.network.loss_per_batch, feed_dict=feed_dict)
                    if temp_optimal_loss < optimal_loss_drift:
                        tqdm.tqdm.write("Loss improved! --- Optimal parameters saved in 'model_drift'")
                        my_network.__save__(sess, model_name='model_drift.ckpt')
                        optimal_loss_drift = temp_optimal_loss


                else:
                    summary_str = summarizer.eval(feed_dict=feed_dict)
                    file_writer.add_summary(summary_str, epoch)
                    temp_optimal_loss = sess.run(my_network.network.loss_per_batch, feed_dict=feed_dict)

                    if temp_optimal_loss < optimal_loss:
                        tqdm.tqdm.write("Loss improved! --- Optimal parameters saved")
                        my_network.__save__(sess)
                        optimal_loss = temp_optimal_loss

        file_writer.close()
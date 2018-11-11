import tensorflow as tf
from tensorflow.contrib import rnn

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Networks():
    def __init__(self, n_dense_layers, n_lstm_layers, units_layer_n_dense, units_layer_n_lstm,
                 logdir, network_type, lag, mean_window, n_output_features=7, n_actions_cols=3, n_state_cols=7):
        self.n_output_features = n_output_features
        self.n_dense_layers = n_dense_layers
        self.n_lstm_layers = n_lstm_layers
        self.units_layer_n_dense = units_layer_n_dense
        self.units_layer_n_lstm = units_layer_n_lstm

        if self.n_dense_layers != len(self.units_layer_n_dense):
            raise ValueError("'n_dense_layers' and 'units_layer_n_dense' don't have matching length")

        if self.n_lstm_layers != len(self.units_layer_n_lstm):
            raise ValueError("'n_lstm_layers' and 'units_layer_n_lstm' don't have matching length")

        self.logdir = logdir
        self.network_type = network_type
        self.lag = lag
        self.mean_window = mean_window
        self.n_actions_cols = n_actions_cols
        self.n_state_cols = n_state_cols
        self.total_time_periods = self.lag + 1  # the number of lag variabels creates the history, but does not account for the current time step
        self.total_input_features = self.mean_window * self.n_actions_cols + self.n_state_cols
        self.network = self.__network__()

    def __create_feed_dict__(self, batch_x, batch_y):
        return {self.network.placeholder_x: batch_x, self.network.target_placeholder: batch_y}

    def __network__(self):
        with tf.name_scope('placeholders'):
            target_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.n_output_features])
            if self.network_type == 'LSTM':
                placeholder_x = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.total_time_periods, self.total_input_features])
                flatten_input_for_lstm = tf.unstack(placeholder_x, self.total_time_periods, 1)
            else:
                placeholder_x = tf.placeholder(dtype=tf.float32,
                                               shape=[None, int(self.total_time_periods * self.total_input_features)])

        with tf.name_scope('layers'):
            lstm_flag = False
            for i in range(self.n_lstm_layers):
                lstm_flag = True
                if i == 0:
                    lstm_layer = tf.nn.rnn_cell.LSTMCell(self.units_layer_n_lstm[i], forget_bias=1,
                                                         name='lstm_unit' + str(i))
                    layer, _ = rnn.static_rnn(lstm_layer, flatten_input_for_lstm, dtype=tf.float32)
                else:
                    lstm_layer = tf.nn.rnn_cell.LSTMCell(self.units_layer_n_lstm[i], forget_bias=1,
                                                         name='lstm_unit' + str(i))
                    layer, _ = rnn.static_rnn(lstm_layer, layer, dtype=tf.float32)

            for i in range(self.n_dense_layers):
                if (i == 0) & (lstm_flag == False):
                    layer = tf.layers.dense(inputs=placeholder_x, units=self.units_layer_n_dense[i],
                                            name='layer' + str(i), activation=tf.nn.relu)
                elif (i == 0) & (lstm_flag == True):
                    layer = tf.layers.dense(inputs=layer[-1], units=self.units_layer_n_dense[i], name='layer' + str(i),
                                            activation=tf.nn.relu)
                else:
                    layer = tf.layers.dense(inputs=layer, units=self.units_layer_n_dense[i], name='layer' + str(i),
                                            activation=tf.nn.relu)

        if (lstm_flag == True) & (self.n_dense_layers > 0):
            prediction = tf.layers.dense(inputs=layer, units=self.n_output_features, name='prediction')
        elif (lstm_flag == True) & (self.n_dense_layers == 0):
            prediction = tf.layers.dense(inputs=layer[-1], units=self.n_output_features, name='prediction')
        else:
            prediction = tf.layers.dense(inputs=layer, units=self.n_output_features, name='prediction')

        loss_per_feature = tf.reduce_mean(tf.squared_difference(prediction, target_placeholder), axis=0,
                                          name='loss_per_feature')
        loss_per_batch = tf.losses.mean_squared_error(labels=target_placeholder, predictions=prediction)
        saver = tf.train.Saver(max_to_keep=10000000)
        return AttrDict(locals())

    def __optimize__(self, optimizer=tf.train.AdamOptimizer()):
        optimizer = optimizer
        minimize = optimizer.minimize(loss=self.network.loss_per_batch)
        return minimize

    def __metrics__(self):
        with tf.name_scope('scalars'):
            m0 = tf.summary.scalar('V_source_mse', self.network.loss_per_feature[0])
            m1 = tf.summary.scalar('I_U_mse', self.network.loss_per_feature[1])
            m2 = tf.summary.scalar('I_V_mse', self.network.loss_per_feature[2])
            m3 = tf.summary.scalar('I_W_mse', self.network.loss_per_feature[3])
            m4 = tf.summary.scalar('sensor_torque_mse', self.network.loss_per_feature[4])
            m5 = tf.summary.scalar('encoder_rpm_mse', self.network.loss_per_feature[5])
            m6 = tf.summary.scalar('temperature_board_mse', self.network.loss_per_feature[6])
            m7 = tf.summary.scalar('total_mse', self.network.loss_per_batch)
            summary = tf.summary.merge_all()
            file_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
        return summary, file_writer

    def __save__(self, sess, model_name='my_model.ckpt'):
        return self.network.saver.save(sess, self.logdir + model_name)

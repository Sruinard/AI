import tensorflow as tf
from tensorflow.contrib import rnn


class Networks():
    def __init__(self, type, n_special_layers, n_dense_layers, n_units, lstm_n_units, Preprocess_input, learning_rate,n_classes=7):
        self.type = type        #choose from ['dense', 'LSTM', 'GRU']
        self.n_special_layers = n_special_layers
        self.n_dense_layers = n_dense_layers
        self.n_units = n_units #provide in list
        self.lstm_n_units = lstm_n_units
        self.preprocess_input = Preprocess_input
        self.n_input = self.preprocess_input.mean_window * 3 + 7    #3 = number of control variables, 7=number of state variables
        self.n_time_steps = (self.preprocess_input.lag_period+1)
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.network = self._network()

    def _network(self):
        if self.type=='dense':
            self.x_placeholder = tf.placeholder(tf.float32, [None, int(self.n_time_steps * self.n_input)])
            # input label placeholder
            self.y_label = tf.placeholder(tf.float32, [None, self.n_classes])

            for i in range(self.n_dense_layers):
                if i == 0:
                    self.layer = tf.layers.dense(units=self.n_units[i], inputs=self.x_placeholder, activation=tf.nn.relu)
                else:
                    self.layer = tf.layers.dense(units=self.n_units[i], inputs=self.layer, activation=tf.nn.relu)


            self.prediction = tf.layers.dense(inputs=self.layer, units=self.n_classes)
            self.loss = tf.losses.mean_squared_error(predictions=self.prediction, labels=self.y_label)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


        if self.type=='LSTM':

            out_weights = tf.Variable(tf.random_normal([self.lstm_n_units, self.n_classes]))
            out_bias = tf.Variable(tf.random_normal([self.n_classes]))

            self.x_placeholder = tf.placeholder(tf.float32, [None, self.n_time_steps, self.n_input])
            # input label placeholder
            self.y_label = tf.placeholder(tf.float32, [None, self.n_classes])

            input_reshaped = tf.unstack(self.x_placeholder, self.n_time_steps, 1)

            # defining the network
            lstm_layer = tf.nn.rnn_cell.LSTMCell(self.lstm_n_units, forget_bias=1)
            outputs, _ = rnn.static_rnn(lstm_layer, input_reshaped, dtype=tf.float32)

            # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
            # prediction=tf.matmul(outputs[-1],out_weights)+out_bias
            self.prediction = tf.layers.dense(outputs[-1], self.n_classes)+out_bias
            # loss_function
            self.loss = tf.losses.mean_squared_error(predictions=self.prediction, labels=self.y_label)
            # optimization
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        #important!!!!! when running LSTM, 'batch_x' has to be reshaped ----> batch_x = np.reshape(np.array(batch_x), newshape=[self.batch_size, int(self.n_time_steps), self.n_input])




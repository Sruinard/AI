import numpy as np
import pandas as pd
import itertools

def generate_names(mean_window):
    col_names_actions = [['switch_U-' + str(i), 'switch_V-' + str(i), 'switch_W-' + str(i)] for i in range(mean_window)]
    col_names_actions = np.reshape(col_names_actions, newshape=-1)
    col_names_states = [['V_source','I_U','I_V','I_W','sensor_torque','encoder_rpm','temperature_board']]
    col_names_states.append(col_names_actions)
    return list(itertools.chain(*col_names_states))

def create_col_names(mean_window, lag):
    col_names_per_time_stamp = generate_names(mean_window)
    col_names = []
    col_names.append(col_names_per_time_stamp)
    for i in range(1, lag + 1):
        col_names_at_time_i = [col_name+'-lag-'+str(i) for col_name in col_names_per_time_stamp]
        col_names.append(col_names_at_time_i)
    return list(itertools.chain(*col_names))

class DriftBatch():
    def __init__(self, batch_x, batch_y, lag, mean_window, batch_size, network_type, decorrelate_flag, network):
        self.batch_x = pd.DataFrame(np.reshape(batch_x, newshape=(batch_size, -1)))
        self.batch_x.columns = create_col_names(lag=lag, mean_window=mean_window)
        self.batch_y = batch_y
        self.lag = lag
        self.mean_window = mean_window
        self.batch_size = batch_size
        self.network_type = network_type
        self.decorrelate_flag = decorrelate_flag
        self.n_action_cols = 3
        self.n_state_cols = 7
        self.network = network.network
        self.length_one_state = self.mean_window * self.n_action_cols + self.n_state_cols
        if self.network_type=='LSTM':
            self.input_state = np.reshape(self.batch_x.iloc[0, :], newshape=(-1, self.lag + 1, self.length_one_state))
        else:
            self.input_state = np.reshape(self.batch_x.iloc[0, :], newshape=(1, -1))

    def create_batch(self, sess, n_drift):
        self.prediction = None
        if n_drift > self.batch_size:
            raise ValueError("'n_drift' MUST BE SMALLER THAN 'batch_size'")

        for i in range(n_drift):
            if i == 0:
                self.input_state = self.input_state
            else:
                self.prediction.sensor_torque = self.batch_y.sensor_torque[i - 1:i].values

                if self.decorrelate_flag:
                    self.prediction.loc[:, self.prediction.columns] += self.batch_x[i - 1:i].loc[:,
                                                                       self.prediction.columns].values

                self.input_state = pd.DataFrame(np.reshape(self.input_state, newshape=(1, -1)))
                self.input_state = self.input_state.iloc[:, :(self.length_one_state) * self.lag]
                self.batch_x[i:i + 1].loc[:, self.prediction.columns] = self.prediction.values
                self.state_actions = self.batch_x[i:i + 1].iloc[:, :(self.length_one_state)]
                self.input_state = np.concatenate((self.state_actions, self.input_state), axis=1)
                if self.network_type=='LSTM':
                    self.input_state = np.reshape(self.input_state, newshape=(-1, self.lag + 1, self.length_one_state))

            feed_dict = {self.network.placeholder_x: self.input_state}
            self.prediction = pd.DataFrame(sess.run(self.network.prediction, feed_dict=feed_dict))
            self.prediction.columns = self.batch_y.columns
        if self.network_type=='LSTM':
            self.batch_x = np.reshape(np.array(self.batch_x),
                                      newshape=[self.batch_size, self.lag + 1, self.length_one_state])

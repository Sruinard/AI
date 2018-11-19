import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataSelection():
    def __init__(self, df, split_percentage,
                 relevant_columns=['V_source', 'switch_U', 'switch_V', 'switch_W', 'I_U', 'I_V', 'I_W', 'sensor_torque',
                                   'encoder_rpm', 'temperature_board']):
        self.relevant_columns = relevant_columns
        self.df = df.loc[:, self.relevant_columns]
        self.split_percentage = split_percentage
        self.train_df, self.val_df = self._create_train_val_data(self.df, self.split_percentage)

    def _create_train_val_data(self, df, split_percentage):
        cut_off_index = int(split_percentage * df.shape[0])
        train_df = df[:cut_off_index]
        val_df = df[cut_off_index:]
        return train_df, val_df


class Preprocessing():
    # class for:
    # - train and val
    # - standardization
    def __init__(self, data_for_fitting,
                 cols_to_standardize=['V_source', 'I_U', 'I_V', 'I_W', 'sensor_torque', 'encoder_rpm',
                                      'temperature_board']):
        self.data_for_fitting = data_for_fitting
        self.cols_to_standardize = cols_to_standardize
        self.fitter = self._fitter(data_for_fitting, self.cols_to_standardize)

    def _fitter(self, data_for_fitting, cols_to_standardize):
        self.cols_to_standardize = cols_to_standardize
        fitter = StandardScaler().fit(self.data_for_fitting.loc[:, self.cols_to_standardize])
        return fitter

    def _transform(self, data_to_transform):
        data_to_transform.loc[:, self.cols_to_standardize] = self.fitter.transform(
            data_to_transform.loc[:, self.cols_to_standardize])
        return data_to_transform

class PreprocessingNormalize():
    # class for:
    # - train and val
    # - standardization
    def __init__(self, data_for_fitting,
                 cols_to_standardize=['V_source', 'I_U', 'I_V', 'I_W', 'sensor_torque', 'encoder_rpm',
                                      'temperature_board']):
        self.data_for_fitting = data_for_fitting
        self.cols_to_standardize = cols_to_standardize
        self.fitter = self._fitter(data_for_fitting, self.cols_to_standardize)

    def _fitter(self, data_for_fitting, cols_to_standardize):
        self.cols_to_standardize = cols_to_standardize
        fitter = MinMaxScaler().fit(self.data_for_fitting.loc[:, self.cols_to_standardize])
        return fitter

    def _transform(self, data_to_transform):
        data_to_transform.loc[:, self.cols_to_standardize] = self.fitter.transform(
            data_to_transform.loc[:, self.cols_to_standardize])
        return data_to_transform


class SpecifyBatching():
    def __init__(self, df, batch_size, neural_network_type, decorrelate=True, skip_n_frames=1, mean_window=1, lag=0,
                 state_cols=['V_source', 'I_U', 'I_V', 'I_W', 'sensor_torque', 'encoder_rpm', 'temperature_board'],
                 action_cols=['switch_U', 'switch_V', 'switch_W']):
        self.df = df
        self.batch_size = batch_size
        self.neural_network_type = neural_network_type
        self.skip_n_frames = np.max((1, skip_n_frames))
        self.mean_window = np.max((1, mean_window))
        self.lag = np.max((0, lag))
        self.state_cols = state_cols
        self.action_cols = action_cols
        self.full_batch_size = (self.batch_size + self.lag) * self.mean_window * self.skip_n_frames
        self.length_one_state = self.mean_window * len(self.action_cols) + len(self.state_cols)
        self.decorrelate = decorrelate

    def _batch_selector(self):
        # accounting for full time periods
        possible_indexes = np.shape(self.df)[0] - (self.mean_window * self.skip_n_frames + self.full_batch_size)
        index = np.random.randint(possible_indexes)
        index_x_batch = index
        index_y_batch = index + (self.mean_window * self.skip_n_frames)
        batch_x = self.df[index_x_batch: index_x_batch + self.full_batch_size]
        batch_y = self.df[index_y_batch: index_y_batch + self.full_batch_size]
        return batch_x, batch_y

    def _skip_frames(self, batch_x, batch_y):
        return batch_x[::self.skip_n_frames], batch_y[::self.skip_n_frames]

    def _mean_window(self, batch_x, batch_y):
        states_x, states_y = batch_x.loc[:, self.state_cols], batch_y.loc[:, self.state_cols]
        states_x = states_x.groupby(np.arange(len(states_x.index)) // self.mean_window, axis=0).mean()
        batch_y = states_y.groupby(np.arange(len(states_y.index)) // self.mean_window, axis=0).mean()

        actions = batch_x.loc[:, self.action_cols]
        actions_taken_within_mean_window = []
        n_chunks = actions.shape[0] / self.mean_window
        for chunk in np.array_split(actions, n_chunks):
            actions_taken_within_mean_window.append(list(reversed(np.reshape(np.array(chunk), newshape=-1))))

        col_names_actions = [['switch_U-' + str(i), 'switch_V-' + str(i), 'switch_W-' + str(i)] for i in
                             range(self.mean_window)]
        col_names_actions = np.reshape(col_names_actions, newshape=-1)
        actions_taken_within_mean_window = pd.DataFrame(actions_taken_within_mean_window, columns=col_names_actions)
        batch_x = pd.concat((states_x, actions_taken_within_mean_window), axis=1)
        return batch_x, batch_y

    def _lag(self, batch_x, batch_y):
        lag_col_names = batch_x.columns
        temp_batch = batch_x
        for i in range(1, self.lag + 1):
            lagged_batch = batch_x.shift(i)
            lagged_batch.columns = [name + '-lag-' + str(i) for name in lag_col_names]
            temp_batch = pd.concat((temp_batch, lagged_batch), axis=1)
        batch_x = temp_batch[self.lag:]
        batch_y = batch_y[self.lag:]
        return batch_x, batch_y

    def _decorrelate(self, batch_x, batch_y):
        batch_y = batch_y - batch_x.loc[:, batch_y.columns]
        return batch_y

    def _input_to_LSTM(self, batch_x):
        batch_x = np.reshape(np.array(batch_x), newshape=[self.batch_size, self.lag + 1, self.length_one_state])
        return batch_x

    def _select_batch(self):
        batch_x, batch_y = self._batch_selector()
        if self.skip_n_frames:
            batch_x, batch_y = self._skip_frames(batch_x, batch_y)
        if self.mean_window:
            batch_x, batch_y = self._mean_window(batch_x, batch_y)
        if self.lag:
            batch_x, batch_y = self._lag(batch_x, batch_y)
        if self.decorrelate:
            batch_y = self._decorrelate(batch_x, batch_y)
        if self.neural_network_type == 'LSTM':
            batch_x = self._input_to_LSTM(batch_x)

        return batch_x, batch_y

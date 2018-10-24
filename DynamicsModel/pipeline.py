import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from sklearn import pipeline
import itertools



class Preprocessing():
    def __init__(self, X_train, X_test, batch_size, skip_n_frames=1, mean_window=1, lag_period=0, keep_all=False, use_default=True, remove_col_flag=True, standardize=True):
        self.X_train = X_train
        self.X_test = X_test
        self.skip_n_frames = np.max((1, skip_n_frames))
        self.mean_window = np.max((1, mean_window))
        self.lag_period = lag_period
        self.batch_size = batch_size * self.skip_n_frames * self.mean_window + np.max((0, self.lag_period)) * self.skip_n_frames * self.mean_window
        self.default_columns = ['V_source', 'I_U', 'I_V', 'I_W', 'sensor_torque', 'encoder_rpm', 'temperature_board']
        self.keep_all = keep_all
        self.use_default = use_default
        self.remove_col_flag = remove_col_flag
        self.standardize = standardize
        if self.remove_col_flag:
            self.X_train = self._remove_control_columns(self.X_train)
            self.X_test = self._remove_control_columns(self.X_test)
        if self.standardize:
            self.X_train = self._standardize(self.X_train)
            self.X_test = self._standardize(self.X_test, validation_flag=True)

    def _remove_control_columns(self, data):
        # USES PANDAS
        if self.keep_all:
            data.loc[:, ['time']] = np.arange(data.shape[0])
        else:
            data = data.loc[:, data.columns[:11]]
            data.loc[:, ['time']] = np.arange(data.shape[0])
            self.all_columns = data.columns
        return data

    def _standardize(self, data, validation_flag=False):
        # USES PANDAS
        if self.use_default == False:
            self.default_columns = list(
                input('which columns do you want to standardize?, type "default" for using the default columns',
                      'choose between:', str(data.columns)))
        if validation_flag == False:
            self.scaler = preprocessing.StandardScaler().fit(self.X_train.loc[:, self.default_columns])
            data.loc[:, self.default_columns] = self.scaler.transform(data.loc[:, self.default_columns])
        if validation_flag == True:
            data.loc[:, self.default_columns] = self.scaler.transform(data.loc[:, self.default_columns])
        return data

    def _history_test_data(self, ):
        # function to add history data to the test set, since it is a time-series dataset
        # Needs modification!
        history_before_test_set = np.array(self.x_train)[-self.batch_size:, :]
        test_data = np.array(self.x_test)
        test_data = np.concatenate(history_before_test_set, test_data)
        return test_data

    def _select_batch(self, data, validation_flag=False, temp_batch_size=None):
        # want a full time_series, so can't use the final indexes in data[-self.batch_size:]
        length_data_set = np.shape(data)[0]
        time_series_suitable_indexes = length_data_set - self.batch_size
        index = np.random.randint(time_series_suitable_indexes)
        x_batch = data[index:index + self.batch_size]
        # target value is next time step, account for skip_n_frames and mean_window
        y_batch = data[index + self.skip_n_frames * self.mean_window:index + self.batch_size + self.skip_n_frames * self.mean_window]
        if validation_flag==True:
            val_batch_size = temp_batch_size * self.skip_n_frames * self.mean_window + np.max((0, self.lag_period)) * self.skip_n_frames * self.mean_window
            time_series_suitable_indexes = length_data_set - val_batch_size
            index = np.random.randint(time_series_suitable_indexes)

            x_batch = data[index:index + val_batch_size]
            # target value is next time step, account for skip_n_frames and mean_window
            y_batch = data[index + self.skip_n_frames * self.mean_window:index + val_batch_size + self.skip_n_frames * self.mean_window]

        return x_batch, y_batch

    def _skip_frames(self, data):
        return data.loc[::self.skip_n_frames, :]

    def _create_flattened_actions(self, batch):
        n_blocks = int(np.shape(batch)[0] / self.mean_window)
        n_cols = self.mean_window * 3
        actions_taken = np.array(batch.loc[:, ['switch_U', 'switch_V', 'switch_W']])
        split_data = np.array_split(actions_taken, n_blocks)
        actions_taken_flattened = np.reshape(split_data, newshape=(n_blocks, n_cols))
        actions_taken_flattened = pd.DataFrame(actions_taken_flattened)
        names_list = [['U_T-' + str(i), 'V_T-' + str(i), 'W_T-' + str(i)] for i in range(self.mean_window)]
        col_names_actions = list(itertools.chain(*names_list))
        actions_taken_flattened.columns = col_names_actions
        return actions_taken_flattened

    def _concatenate(self, rolling_mean_data, flattened_actions_data):
        concatenated_data = pd.concat((rolling_mean_data, flattened_actions_data), axis=1)
        return concatenated_data

    def _rolling_mean(self, batch, y_flag=False):
        # USES PANDAS
        if y_flag==False:
            flattened_actions = self._create_flattened_actions(batch)
            batch = batch.loc[:, self.default_columns].groupby(np.arange(len(batch.index)) // self.mean_window,
                                                               axis=0).mean()
            batch = self._concatenate(batch, flattened_actions)
        if y_flag==True:
            batch = batch.loc[:, self.default_columns].groupby(np.arange(len(batch.index)) // self.mean_window,
                                                               axis=0).mean()
        return batch

    def _create_lag(self, batch):
        history_data = []
        batch_temp = batch[self.lag_period:]
        index_values = batch_temp.index.values
        for i in range(len(batch_temp)):
            look_back = batch[i:(i + self.lag_period)]
            look_back = np.reshape(np.array(look_back), newshape=(-1))
            history_data.append(look_back)
        history_data = pd.DataFrame(history_data)
        history_data.reset_index(drop=True, inplace=True)
        batch_temp.reset_index(drop=True, inplace=True)
        lagged_data = pd.concat((batch_temp, history_data), axis=1)
        lagged_data = lagged_data.set_index(index_values)
        return lagged_data

    def _preprocess(self, skip_frames=True, mean_window=True, validation_flag=False, temp_batch_size=None, decorrelate=True):
        x_batch, y_batch = self._select_batch(self.X_train)
        if validation_flag:
            x_batch, y_batch = self._select_batch(self.X_test, validation_flag=validation_flag, temp_batch_size=temp_batch_size)

        if skip_frames == True:
            x_batch = self._skip_frames(data=x_batch)
            y_batch = self._skip_frames(data=y_batch)
        if mean_window == True:
            x_batch = self._rolling_mean(x_batch)
            y_batch = self._rolling_mean(y_batch, y_flag=True)
        x_batch = self._create_lag(x_batch)
        y_batch = self._create_lag(y_batch)
        if decorrelate:
            y_batch.loc[:, self.default_columns] = y_batch.loc[:, self.default_columns] - x_batch.loc[:,self.default_columns]
        return x_batch, y_batch.loc[:, self.default_columns]


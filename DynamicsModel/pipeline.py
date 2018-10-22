import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from sklearn import pipeline

data = pd.read_csv('/Users/stefruinard/Documents/ML6/DataECC/exp3_010.csv')

x_train = data.loc[:1000, :]
x_test = data.loc[1000:1500, :]


class Preprocessing():
    def __init__(self, X_train, X_test, batch_size, history_length, skip_n_frames=1, time_window=1):
        self.X_train = X_train
        self.X_test = X_test
        self.skip_n_frames = skip_n_frames
        self.time_window = time_window
        self.batch_size = batch_size * np.max()
        self.history_length = history_length
        self.default_columns = ['V_source', 'I_U', 'I_V', 'I_W', 'sensor_torque', 'encoder_rpm', 'temperature_board']

    def _remove_control_columns(self, data, keep_all=False):
        # USES PANDAS
        if keep_all:
            data.loc[:, ['time']] = np.arange(data.shape[0])
        else:
            data = data.loc[:, data.columns[:11]]
            data.iloc[:, ['time']] = np.arange(data.shape[0])
        return data

    def _standardize(self, data, use_default=True):
        #USES PANDAS
        if use_default==False:
            self.default_columns = list(input('which columns do you want to standardize?, type "default" for using the default columns', 'choose between:', str(data.columns)))

        scaler = preprocessing.StandardScaler(self.X_train.loc[:, self.default_columns])
        data.loc[:,default_columns] = scaler.transform(data.loc[:, self.default_columns])
        return data

    def _history_test_data(self):
        #function to add history data to the test set, since it is a time-series dataset
        self.n_rows_to_append = self.skip_n_frames * self.time_window
        history_before_test_set = np.array(self.x_train)[-self.n_rows_to_append:,:]
        test_data = np.array(self.x_test)
        test_data = np.concatenate(history_before_test_set, test_data)
        return test_data

    def _skip_frames(self, skip_n_frames, data):
        self.skip_n_frames = skip_n_frames
        return np.array(data)[::skip_n_frames,:]

    def _prepare_rolling_mean(self, time_window, data, is_training):
        #handle corner cases when data%time_window !=0
        #Returns Pandas DF
        columns = data.columns
        rows_in_last_section = len(data.index)%time_window
        data_start = np.array(data[:-rows_in_last_section])
        data_last_full_time_window = np.array(data[-time_window:])
        data = pd.DataFrame(np.concatenate((data_start, data_last_full_time_window)))
        data.columns = columns
        return data

    def _create_flattened_actions(self, prepared_rolling_mean_data):
        prepared_data = self._prepare_rolling_mean(self.time_window, prepared_rolling_mean_data)
        n_blocks = int(np.shape(prepared_data)[0]/self.time_window)
        n_cols = time_window*3
        actions_taken = np.array(prepared_data.loc[:, ['switch_U','switch_V', 'switch_W']])
        split_data = np.array_split(actions_taken, n_blocks)
        actions_taken_flattened = np.reshape(split_data, newshape=(n_blocks, n_cols))
        return actions_taken_flattened


    def _rolling_mean(self, time_window, data):
        # USES PANDAS
        self.time_window = time_window
        data = data.loc[:,self.default_columns].groupby(np.arange(len(data.index))//time_window, axis=0).mean()
        return data

    def concatenate(self, rolling_mean_data, flattened_actions_data):
        rolling_mean_data = np.array(rolling_mean_data)
        flattened_actions_data = np.array(flattened_actions_data)
        concatenated_data = np.concatenate((rolling_mean_data,flattened_actions_data), axis=1)
        return concatenated_data

    def _select_batch(self, data):
        # want a full time_series, so can't use the final indexes in data[-self.batch_size:]
        length_data_set = np.shape(data)[0]
        time_series_suitable_indexes = length_data_set - self.batch_size
        index = np.random.randint(time_series_suitable_indexes)
        batch = data[index:index+self.batch_size, :]
        return batch

    def _to_time_series(self,):
        print('')




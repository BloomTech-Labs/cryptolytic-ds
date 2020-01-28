# begin imports
from __future__ import print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
# from scipy.stats import yeojohnson
import os
from cryptolytic.util import *

# tensorflow import
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Activation,\
    Add, Input, LSTM
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.constraints as constraints
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.initializers import glorot_uniform, glorot_uniform

# sklearn imports
from sklearn.preprocessing import MinMaxScaler


def thing(arg, axis=0):
    x = np.sign(arg) * np.log(np.abs(arg) + 1)
    mu = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    return x, mu, std


# normalization version 2
def normalize(df):
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        df = df.values
    if np.ndim(df) == 1:
        df = np.expand_dims(df, axis=1)
    df = df.copy()
    x, mu, std = thing(df, axis=0)
    for i in range(df.shape[1]):
        df[:, i] = (x[:, i] - mu[i]) / std[i]
    return df


# denormalization version 2
def denormalize(values, df, col=None):
    values = values.copy()

    def eq(x, mu, std):
        return np.exp((x*std) + mu) - 1

    if np.ndim(values) == 1 and col is not None:
        x, mu, std = thing(df[col])
        return eq(values, mu, std)
    else:
        for i in range(values.shape[1]):
            x, mu, std = thing(df.iloc[:, i])
            if isinstance(values, pd.DataFrame):
                values.iloc[:, i] = eq(values.iloc[:, i], mu, std)
            else:
                values[:, i] = eq(values[:, i], mu, std)
    return values


def windowed(dataset, target, batch_size, history_size, step, lahead=1, ratio=0.8):
    xs = []
    ys = []

    x = dataset
    y = dataset[:, target]

    start = history_size  # 1000
    end = dataset.shape[0] - lahead  # 4990
    # 4990 - 1000 = 3990
    for i in range(start, end):
        # grab rows from i, to i+history_size
        indices = range(i-history_size, i, step)
        xs.append(x[indices])
        ys.append(y[i:i+lahead])

    xs = np.array(xs)
    ys = np.array(ys)

    nrows = xs.shape[0]
    train_size = int(nrows * ratio)
    # make sure the sizes are multiples of the batch size (needed for stateful
    # lstm)
    train_size -= train_size % batch_size
    val_size = nrows - train_size
    val_size -= val_size % batch_size
    total_size = train_size + val_size
    xs = xs[:total_size]
    ys = ys[:total_size]

    return xs[:train_size], ys[:train_size], xs[train_size:], ys[train_size:]


# define function to load data for training and validation
def data_load():
    # load data
    TRAIN_SPLIT = 3000
    # dataset = normalize_df(df._get_numeric_data()).values
    dataset = normalize_df(df._get_numeric_data()).values
    target = df.columns.get_loc('close') - 1
    y = dataset[:, target]
    past_history = 720
    future_target = 5
    STEP = 3

    x_train, y_train = multivariate_data(dataset, y, 0,
                                         TRAIN_SPLIT, past_history,
                                         future_target, STEP,
                                         single_step=False)
    x_val, y_val = multivariate_data(dataset, y,
                                     TRAIN_SPLIT, None, past_history,
                                     future_target, STEP,
                                     single_step=False)
    print('Single window of past history : {}'.format(x_train[0].shape))
    print('\n Target temperature to predict : {}'.format(y_train[0].shape))
    BUFFER_SIZE = 10_000
    BATCH_SIZE = 256
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).\
        batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    '''
    code for data loading here
    '''
    return train_data, val_data

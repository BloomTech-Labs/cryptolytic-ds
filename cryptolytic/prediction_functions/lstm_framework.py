# begin imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from __future__ import print_function

# tensorflow import
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM

# configure the parameters
# LSTM variables
lstm_output_dimensionality = 1
# data slicing
buffer_size = 10000
batch_size = 32
# model fit variables
epochs = 10
evaluation_interval = 200
validation_steps = 50

# load data
'''
code for data loading here
'''

# build the framework
# Sequential network
framework_lstm = Sequential()
# add lstm layers to the framework
# input shape not yet added as import method not yet known
framework_lstm.add(LSTM(lstm_output_dimensionality,
                        input_shape=(x_train.shape[])))
# add dense layers to the framework
framework_lstm.add(Dense(1))

# compile the framework
# default optimizer adam, more research required for optimal optimizer
# default loss binary_crossentropy, more research required for optimal loss
# default metrics accuracy, more research required for what we actually want
# the model to predict
framework_lstm.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

# slice the train and validation data to later fit the model
train_univariate = tf.data.Dataset.from_tensor_slices(
    ('''x_train data''', '''y_train data'''))
train_univariate = train_univariate.cache().shuffle(buffer_size)\
    .batch(batch_size).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices(
    ('''x_val data''', '''y_val data'''))
val_univariate = val_univariate.batch(batch_size).repeat()

# fit the model
framework_lstm.fit(train_univariate,
                   epochs=epochs,
                   steps_per_epoch=evaluation_interval,
                   validation_data=val_univariate,
                   validation_steps=validation_steps)

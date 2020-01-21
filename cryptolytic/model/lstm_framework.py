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


# define function to load data for training and validation
def data_load():
    # load data
    '''
    code for data loading here
    '''
    return data


def create_framework_lstm(x_train, lstm_output_dimensionality=1):
    '''
    Function to build framework of model
    Requires input of x_train, output dimensionality defaulted to 1
    Returns a framework model
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
    # return the created framework
    return framework_lstm


def model_compile(model_framework,
                  loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']):
    '''
    Compile the framework
    Requires input of a framework model
    Default optimizer adam, more research required for optimal optimizer
    Default loss binary_crossentropy, more research required for optimal loss
    Default metrics accuracy, more research required for what we actually want
    the model to predict
    Returns compiled model
    '''
    compiled_model = model_framework
    compiled_model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
    return compiled_model


def slice_data(x_train, y_train, x_val, y_val, batch_size=32):
    '''
    Define funciton for data slicing
    Requires input data of x_train, y_train, x_val, y_val
    Defaults batch_size
    Returns train_univariate and val_univariate
    '''
    # slice the train and validation data to later fit the model
    train_univariate = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    train_univariate = train_univariate.cache().batch(batch_size).repeat()
    val_univariate = tf.data.Dataset.from_tensor_slices(
        (x_val, y_val))
    val_univariate = val_univariate.repeat()
    # return the sliced train and validation data
    return train_univariate, val_univariate


def model_fit(compiled_model,
              train_univariate,
              val_univariate,
              epochs=10,
              steps_per_epoch=200,
              validation_steps=50):
    '''
    Funciton to fit the model
    Requires a compiled model, train data, validation data
    Defaults for epochs, steps_per_epoch, and valitation_steps
    Returns fitted model
    '''
    fit_model = compiled_model
    fit_model.fit(train_univariate,
                  epochs=epochs,
                  steps_per_epoch=evaluation_interval,
                  validation_data=val_univariate,
                  validation_steps=validation_steps)
    return fit_model

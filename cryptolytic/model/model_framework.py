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
import tensorflow.keras.layers as layers


def normalize_df(df):
    df = df.copy()
    df = df._get_numeric_data()
    # normalize each point by dividing by the inital point and subtracting 1
    for col in df.columns:
        df[col] = (df[col] / (df[col][0] + 1e-5)) - 1
    return df


def denormalize_results(close_preds):
    return (1 + close_preds) * df.close[0]


# should rename
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


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


# add model params
def lstm_model():
    # future_target = 5
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(128, activation='relu'))
    model.add(layers.Dense(future_target))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss='mse')
    return model


# move to plot
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def predictions():
    preds = denormalize_results(model.predict(x_train))
    rcParams['figure.figsize'] = 20, 3
    plt.plot(np.arange(2000), d.denoise(actual[past_history:2000+past_history],
                                        5), label='actual')
    plt.plot(range(2000), d.denoise(preds[:, 0][:2000], 5), label='predicted')
    plt.legend()
    return preds


def fit_model():

    model = lstm_model(x_train)

    history = model.fit(train_data,
                        epochs=9,
                        steps_per_epoch=38,
                        use_multiprocessing=True,
                        workers=4,
                        validation_data=val_data,
                        validation_steps=5)  # (x_val, y_val))


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
    Function to fit the model
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

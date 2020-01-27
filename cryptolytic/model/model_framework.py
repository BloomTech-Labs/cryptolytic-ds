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


def normalize_df(df):
    df = df.copy()
    df = df._get_numeric_data()
    # normalize each point by dividing by the inital point and subtracting 1
    for col in df.columns:
        df[col] = (df[col] / (df[col][0] + 1e-5)) - 1
    return df


def denormalize_results(close_preds):
    return (1 + close_preds) * df.close[0]


# normalization version 2
def normalize(df):
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        df = df.values
    if np.ndim(df) == 1:
        df = np.expand_dims(df, axis=1)
    df = df.copy()
    x, mu, std = thing(df, axis=0)
    for i in range(df.shape[1]):
        df[:, i] = (x[:, i] - mu[i]) / stf[i]
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


def windowed(df, target, batch_size, history_size, step, lahead=1, ratio=0.8):
    xs = []
    ys = []

    x = dataset
    y = dataset[:, target]

    start = history_size # 1000
    end = df.shape[0] - lahead # 4990
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
    # make sure the sizes are multiples of the batch size (needed for stateful lstm)
    train_size -= train_size % batch_size
    val_size = nrows - train_size
    val_size -= val_size % batch_size
    total_size = train_size + val_size
    xs = xs[:total_size]
    ys = ys[:total_size]

    return xs[:train_size], ys[:train_size], xs[train_size:], ys[train_size:]


def conv_block(X, filters, f, stage, block, strides=2):
    """
    X -- input tensor
    f -- interger specify shape of conv window
    stage - integer, used to named layers, depending on position in network.
    stride -- stride
    """
    conv_base_name = 'conv' + str(stage) + block
    bn_base_name = 'bn' + str(stage) + block
    
    f1, f2, f3 = filters
    
    X_shortcut = X
    X = Conv1D(f1, kernel_size=1, strides=strides, name=conv_base_name + '_1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)
    
    X = Conv1D(f2, kernel_size=f, strides=1, padding='same', name=conv_base_name + '_2', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)
    
    X = Conv1D(f3, kernel_size=1, strides=1, padding='valid', name=conv_base_name + '_3', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    # skip connection
    X_shortcut = Conv1D(f3, kernel_size=1, strides=strides, padding='valid', name=conv_base_name + '_4', 
               kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    
    assert tf.keras.backend.int_shape(X) == tf.keras.backend.int_shape(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)

    return X


def identity_block(X, f, filters, stage, block, strides=2):
    """
    X -- input tensor
    f -- interger specify shape of conv window
    stage - integer, used to named layers, depending on position in network.
    stride -- stride
    """
    conv_base_name = 'conv' + str(stage) + block
    bn_base_name = 'bn' + str(stage) + block
    
    f1, f2, f3 = filters
    
    X_shortcut = X
    X = Conv1D(f1, kernel_size=1, strides=strides, name=conv_base_name + '_1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)
    
    X = Conv1D(f2, kernel_size=f, strides=1, padding='same', name=conv_base_name + '_2', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)
    
    X = Conv1D(f3, kernel_size=1, strides=strides, padding='valid', name=conv_base_name + '_3', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    # skip connection, without also convolving x_shortcut
    assert tf.keras.backend.int_shape(X) == tf.keras.backend.int_shape(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)
    return X


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


# def fit_model():

#     model = lstm_model(x_train)

#     history = model.fit(train_data,
#                         epochs=9,
#                         steps_per_epoch=38,
#                         use_multiprocessing=True,
#                         workers=4,
#                         validation_data=val_data,
#                         validation_steps=5)  # (x_val, y_val))


def fit_model(model, inputX, inputy):
    epochs = 5
    # batch size higher than 1 c  epochs = 10
    for i in range(epochs):
        model.fit(inputX,
                  inputy, 
                  batch_size=batch_size,
                  epochs=1,
                  verbose=1,
                  use_multiprocessing=True,
                  shuffle=False,
                  workers=4,
                  validation_data = (x_val, y_val))
        history['loss'].append(model.history.history['loss'])
        history['val_loss'].append(model.history.history['val_loss'])
#        model.reset_states()
    #pred = transformer.denormalize(model.predict(x_val)[:, 0], df, 'close')
    #pred_history.append(pred)
        
    return model
    
def save_model(model, folder, params=None):
    path = f'models/{folder}'
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, 'model_' + str(np.random.rand()) + '.h5')
    if os.path.exists(filename):
        return save_model(model)
    if params is not None:
        param_path = os.path.join(path, 'model_params.csv')
        pd.DataFrame(params).to_csv(param_path)
        
    model.save(filename)
    print('Saved model')
    
def load_all_models(folder):
    models = []
    params = []
    path = f'models/{folder}'
    for m in os.listdir(path):
        if m.endswith('.csv'):
            params.append(pd.read_csv(m))
        if not m.endswith('.h5'):
            continue
        m = tf.keras.models.load_model(m)
        models.append(m)
        print('Loaded %s' % path)
    return models, params

def stacked_dataset(models, inputX):
    stackX = None
    for model in models:
        yhat = model.predict(inputX, verbose=0)
        #stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilties]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

def fit_stacked_model(models, inputX, inputy):
    stackedX = stacked_dataset(models, inputX)
    # fit model
    model = create_model(inputX)
    fit_model(model, inputX, inputy)
    return model

def stacked_prediction(models, model, inputX):
    stackedX = stacked_dataset(models, inputX)
    yhat = model.predict(stackedX)
    return yhat

def endingly():
    models, params = load_all_models('example')
    model = fit_stacked_model(models, x_train, y_train)
    preds = stacked_prediction(models, model, x_train)
    evaluate_models(models)
    return preds

def hyperparameter(inputX, inputy):
    filtershape1 = 32 + 16 * np.random.randint(0, 6)
    filtershape2 = 32 + 16 * np.random.randint(0, 6)
    
    params = adict(
        filters1 = 32 + 16 * np.random.randint(0, 5),
        noise1 = np.random.uniform(high=0.01),
        filtershape1 = [filtershape1, filtershape1, filtershape1*2],
        filtershape2 = [filtershape2, filtershape2, filtershape2*2]
    )
    print(params)
    model = create_model(inputX, params)
    fit_model(model, inputX, inputy)
    save_model(model, 'filter', params=params)
    return model
                 
def run_tuning():
    models = []
    for i in range(50):
        models.append(hyperparameter(x_train, y_train))
    return models

   
def evaluate_models(models):
    for m in models:
        _, acc = model.evaluate(x_val, y_val)


def create_model(x_train, params):
    attention_size = 5
    input_shape = x_train.shape[-2:]
    X_input = Input(input_shape, batch_size=batch_size)
    X = X_input
    X = layers.ZeroPadding1D(padding=2)(X)
#    X = layers.TimeDistributed(Dense(input_shape[-1], kernel_constraint=constraints.max_norm(1.0), activation='tanh'))(X)
    X = Conv1D(filters=params.filters1, kernel_size=6, strides=1, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.GaussianNoise(params.noise1)(X)
    #X = layers.AveragePooling1D(2, strides=1)(X)
    X = conv_block(X, f=3, filters=params.filtershape1, stage=2, block='a', strides=2)
    X = layers.GaussianNoise(params.noise1)(X)
    X = identity_block(X, f=3, filters=params.filtershape1, stage=2, block='b', strides=1)
    X = layers.GaussianNoise(params.noise1)(X)
    X = identity_block(X, f=3, filters=params.filtershape1, stage=2, block='c', strides=1)
    
    print(X.get_shape())
    print(X_input.get_shape())
    #X = layers.concatenate([X, X_input])
    
   # X = layers.AveragePooling1D(2, name="avg_pool")(X)
    
#    rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(2)]
#    stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
#    lstm_layer = tf.keras.layers.RNN(stacked_lstm)
#    X = lstm_layer(X)
 
    X = layers.Flatten()(X)
    X = Dense(256)(X)
    X = Dense(lahead, kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input, outputs=X, name='model1')
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001))
    return model


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
                            input_shape=(x_train.shape)))
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


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    
    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
        label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
            label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def plot_train_history(loss, val_loss, title):
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    
    plt.show()


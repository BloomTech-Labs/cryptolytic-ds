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
    X = Conv1D(f1, kernel_size=1, strides=strides, name=conv_base_name + '_1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)

    X = Conv1D(f2, kernel_size=f, strides=1, padding='same',
               name=conv_base_name + '_2',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)

    X = Conv1D(f3, kernel_size=1, strides=1, padding='valid',
               name=conv_base_name + '_3',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # skip connection
    X_shortcut = Conv1D(f3, kernel_size=1, strides=strides, padding='valid',
                        name=conv_base_name + '_4',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    assert tf.keras.backend.int_shape(X) == tf.keras.backend.\
        int_shape(X_shortcut)
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
    X = Conv1D(f1, kernel_size=1, strides=strides, name=conv_base_name + '_1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)

    X = Conv1D(f2, kernel_size=f, strides=1, padding='same',
               name=conv_base_name + '_2',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('tanh')(X)

    X = Conv1D(f3, kernel_size=1, strides=strides, padding='valid',
               name=conv_base_name + '_3',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # skip connection, without also convolving x_shortcut
    assert tf.keras.backend.int_shape(X) == tf.keras.backend.\
        int_shape(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('tanh')(X)
    return X


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


def predictions():
    preds = denormalize_results(model.predict(x_train))
    rcParams['figure.figsize'] = 20, 3
    plt.plot(np.arange(2000), d.denoise(actual[past_history:2000+past_history],
                                        5), label='actual')
    plt.plot(range(2000), d.denoise(preds[:, 0][:2000], 5), label='predicted')
    plt.legend()
    return preds


def fit_model(model, inputX, inputy, x_val, y_val, batch_size=200):
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
                  validation_data=(x_val, y_val))
        # history['loss'].append(model.history.history['loss'])
        # history['val_loss'].append(model.history.history['val_loss'])
#        model.reset_states()
    # pred = transformer.denormalize(model.predict(x_val)[:, 0], df, 'close')
    # pred_history.append(pred)

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


def fit_stacked_model(models, inputX, inputy):
    stackedX = stacked_dataset(models, inputX)
    # fit model
    model = create_model(inputX)
    fit_model(model, inputX, inputy)
    return model


def create_model(x_train, params, batch_size=200, lahead=12*3, ):
    attention_size = 5
    input_shape = x_train.shape[-2:]
    X_input = Input(input_shape, batch_size=batch_size)
    X = X_input
    X = layers.ZeroPadding1D(padding=2)(X)
#    X = layers.TimeDistributed(Dense(input_shape[-1],
#                               kernel_constraint=constraints.max_norm(1.0),
#                               activation='tanh'))(X)
    X = Conv1D(filters=32, kernel_size=6, strides=1, name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.GaussianNoise(.005)(X)
    # X = layers.AveragePooling1D(2, strides=1)(X)
    X = conv_block(X, f=3, filters=[48, 48, 96], stage=2, block='a',
                   strides=2)
    X = layers.GaussianNoise(.005)(X)
    X = identity_block(X, f=3, filters=[48, 48, 96], stage=2, block='b',
                       strides=1)
    X = layers.GaussianNoise(.005)(X)
    X = identity_block(X, f=3, filters=[48, 48, 96], stage=2, block='c',
                       strides=1)

    print(X.get_shape())
    print(X_input.get_shape())
    # X = layers.concatenate([X, X_input])

    # X = layers.AveragePooling1D(2, name="avg_pool")(X)

#    rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(2)]
#    stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
#    lstm_layer = tf.keras.layers.RNN(stacked_lstm)
#    X = lstm_layer(X)

    X = layers.Flatten()(X)
    X = Dense(256)(X)
    X = Dense(lahead, kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='model1')
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001))
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


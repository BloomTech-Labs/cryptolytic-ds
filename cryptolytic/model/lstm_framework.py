# Perform imports

# Internal Imports
import cryptolytic.util as util
import cryptolytic.start as start
import cryptolytic.viz.plot as plot
import cryptolytic.data.sql as sql
import cryptolytic.data as d
from cryptolytic.util import *
import cryptolytic.data.historical as h
import cryptolytic.model as m

# External general imports
import pandas as pd
import numpy as np
from matplotlib.pylab import rcParams
# to stop a warning message
from pandas.plotting import register_matplotlib_converters

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras.layers as layers

ohclv = ['open', 'high', 'close', 'low', 'volume']


def feature_engineering(df):
    '''
    Function takes in raw candlestick data dateframe and engineers the data
    for the model
    Requires raw candlestick data dataframe
    Returns Engineered dataframe
    '''
    # Create useful data from important data
    c = df[['close', 'volume', 'diff', 'arb_signal']]
    a_df = c.rolling(6).mean().bfill().rename(columns=lambda x: x+'_mean')
    b_df = c.rolling(6).std().bfill().rename(columns=lambda x: x+'_std')
    c_df = c.rolling(6).skew().bfill().rename(columns=lambda x: x+'_skew')
    d_df = c.rolling(6).kurt().bfill().rename(columns=lambda x: x+'_kurt')
    # Concatinate the created data from above to original dataframe
    df = pd.concat([df, a_df, b_df, c_df, d_df], axis=1).dropna(axis=1)
    # Drop not needed dataframe columns
    df_sub = df.drop(['timestamp', 'period', 'open', 'high', 'low', 'api',
                      'exchange', 'trading_pair'], axis=1)
    return df


def normalize_df(df):
    '''
    Function normalizes the redata before input into the model
    Requires dataframe of data
    Returns normalized dataframe of data
    '''
    df = df.copy()
    df = df._get_numeric_data()
    # normalize each point by dividing by the inital point and subtracting 1
    for col in df.columns:
        df[col] = (df[col] / (df[col][0] + 1e-5)) - 1
    return df


def denormalize_results(close_preds):
    '''
    Fuction undoes the normalization to see actual predicted values
    Requires normalized model output
    Returns denormalized model output
    '''
    return (1 + close_preds) * df.close[0]


def multivariate_data(dataset, target, start_index, end_index=None,
                      history_size=720, target_size=5, step=3,
                      single_step=False):
    '''
    Function creates multivaiate x and y data from normalized dataframe
    Requires dataframe, target column, start_index, end_index
    Defaults for history_size, target_size, step, single_step
    Returns x and y data
    '''
    # Initialize the lists of column names
    data = []
    labels = []

    # Create the end_index if end_index is the default of None
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    # Find the x and y values based on the column indexes and stepd
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    # Return the data as x, y
    return np.array(data), np.array(labels)


def lstm_model(train, future_target=5):
    '''
    Function creates a LSTM model with train data
    Requires train data
    Returns (unfit) model
    '''
    # Model base is Tenserflow Keras Sequential model
    model = tf.keras.models.Sequential()
    # Create Dense model layer
    model.add(layers.Dense(128, use_bias=False))
    # Create Normalization layer
    model.add(layers.LayerNormalization())
    # Create Parametric Rectified Linear Unit layer
    model.add(layers.PReLU())
    # Create Dropout layer
    model.add(layers.Dropout(0.20))
    # Create LSTM layer
    model.add(layers.LSTM(128, return_sequences=True))
    # Create LSTM layer with Rectified Linear Unit activation
    model.add(layers.LSTM(128, activation='relu'))
    # Create Dense layer
    model.add(layers.Dense(future_target))
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss='mse')
    # Return compiled model
    return model


def fit_model(model, train_data, validation_data, epochs=9, steps_per_epoch=38,
              use_multiprocessing=True, workers=4, validation_steps=5):
    '''
    Function fits a compiled model with training and validation data
    Requires compiled model, training data, validation data
    Defaults for epochs, steps per epoch, use of multoprocessing, number of
    workers, and validation steps
    Returns fit model
    '''
    model.fit(train_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              use_multiprocessing=use_multiprocessing,
              workers=workers,
              validation_data=validation_data,
              validation_steps=validation_steps)
    return model


def multi_step_plot(history, true_future, prediction):
    '''
    '''
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


def plot_train_history(history, title):
    '''
    Creates plot based on the fit model
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def fit_model_from_raw_data(df, train_split=3000, history_size=720,
                            target_size=5, step=5, single_step=False,
                            buffer_size=10000, batch_size=256, model_epochs=9,
                            steps_per_epoch=38, use_multiprocessing=True,
                            workers=4, validation_steps=5):
    '''
    Creates a model from the raw data using above functions as base
    Requires raw candlestick data
    Returns fit_model, x_train, y_train, x_val, y_val, engineered datafram
    '''

    # Feature engineer the data
    df = feature_engineering(df)

    # Normalize the numerical data in dataset
    dataset_normalized = normalize_df(df._get_numeric_data()).values

    # Get target column for the model
    target = df.columns.get_loc('close') - 1
    # Get values for target column
    y = dataset_normalized[:, target]

    # Get x and y train and validation sets from normalized dataset
    x_train, y_train = multivariate_data(
        dataset=dataset_normalized,
        target=y,
        start_index=0,
        end_index=train_split,
        history_size=history_size,
        target_size=target_size,
        step=step,
        single_step=single_step)

    x_val, y_val = multivariate_data(
        dataset=dataset_normalized,
        target=y,
        start_index=train_split,
        end_index=None,
        history_size=history_size,
        target_size=target_size,
        step=step,
        single_step=single_step)

    # Create train data and validation data from x and y train and validate
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(buffer_size).\
        batch(batch_size).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(batch_size).repeat()

    # Create the LSTM model
    model = lstm_model(x_train, future_target=target_size)

    # Fit the LSTM model
    history = fit_model(
        model=model,
        train_data=train_data,
        validation_data=val_data,
        epochs=model_epochs,
        steps_per_epoch=steps_per_epoch,
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        validation_steps=validation_steps)

    # Return fit model, x_train, y_train, x_val, y_val
    return history, x_train, y_train, x_val, y_val, df


def predict_on_training_data(model, x_train, df):
    '''
    Function predicts denormalized based on fitted model, training data,
    and actual results
    Requires fitted model, data predicted on
    Outputs predicted values and shows plot
    '''
    # Get predictions from model and actual values from dataframe
    preds = denormalize_results(model.predict(x_train))
    actual = df.close

    # Configure the graph
    rcParams['figure.figsize'] = 20, 3
    plt.plot(np.arange(2000), d.denoise(actual[past_history:2000+past_history],
                                        5), label='actual')
    plt.plot(range(2000), d.denoise(preds[:, 0][:2000], 5), label='predicted')
    plt.legend()

    # Plot the model
    plt.title('All predictions')
    plt.plot(range(2000), d.denoise(preds[:2000], 5))

    return preds


def predict_on_validation_data(model, x_val, y_val):
    '''
    Function predicts denormalized based on fitted model, validation data
    '''
    # Get predictions from model and actual results from validation data
    val_preds = denormalize_results(model.predict(x_val))
    val_actual = denormalize_results(y_val[:, 0])

    # Configure and plot the graph
    plt.plot(np.arange(2000), d.denoise(
        val_actual[past_history:2000+past_history], 20), label='actual')
    plt.plot(range(2000), d.denoise(
        val_preds[:, 0][:2000], 20), label='predicted')
    plt.legend()

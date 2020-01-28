# begin imports

# begin internal imports
import cryptolytic.model.model_framework as model_framework
import cryptolytic.model.data_work as dw
import cryptolytic.data as d
import cryptolytic.model.model_framework as mfw

# tensorflow imports
import tensorflow as tf

# begin general external imports
import ta
import pandas as pd
import time


def run_model():
    # use default params
    now = time.time()
    now = int(now)
    params = {'filters1': 32, 'noise1': 0.007044669933974564,
              'filtershape1': [48, 48, 96], 'filtershape2': [64, 64, 128]}
    history_size = 400
    input_len = 16000
    lahead = 12*3
    step = 2
    period = 300
    to_drop = lahead - 1
    input_len = input_len + to_drop
    now = now - input_len*period
    batch_size = 200

    # create base dataframe
    df_orig = None
    df_orig = d.get_df(
        {'start': now, 'period': period,
         'trading_pair': 'btc_usd', 'exchange_id': 'bitfinex'},
        n=input_len
        )

    # Perform work on the dataframe
    df = df_orig
    df = df.sort_index()
    df = df._get_numeric_data().drop(["period"], axis=1, errors='ignore')
    # filter out timestapm_ metrics
    df = df.filter(regex="(?!timestamp_.*)", axis=1)
    df = ta.add_all_ta_features(df, open="open", high="high", low="low",
                                close="close", volume="volume").dropna(axis=1)
    df_diff = (df - df.shift(1, fill_value=0)).\
        rename(lambda x: x+'_diff', axis=1)
    df = pd.concat([df, df_diff], axis=1)
    dataset = dw.normalize(df.values)
    target = df.columns.get_loc('close')
    y = dataset[:, target]
    history = {'loss': [], 'val_loss': []}

    # create the x and y train and val data
    x_train, y_train, x_val, y_val = dw.windowed(
        dataset, target, batch_size, history_size, step, lahead
        )

    # TODO
    # load a created model to fit upon load with updated data

    # create the model
    model = mfw.create_model(x_train, params)

    # fit the model
    model = mfw.fit_model(model, x_train, y_train, x_val, y_val)

    mfw.save_model(model, 'models')


def cron_pred():
    path = 'models/models/model_eth_btc.h5'
    model = tf.keras.models.load_model(path)

    # use default params
    now = time.time()
    now = int(now)
    params = {'filters1': 32, 'noise1': 0.007044669933974564,
              'filtershape1': [48, 48, 96], 'filtershape2': [64, 64, 128]}
    history_size = 400
    input_len = 16000
    lahead = 12*3
    step = 2
    period = 300
    to_drop = lahead - 1
    input_len = input_len + to_drop
    now = now - input_len*period
    batch_size = 200

    # create base dataframe
    df_orig = None
    df_orig = d.get_df(
        {'start': now, 'period': period,
         'trading_pair': 'btc_usd', 'exchange_id': 'bitfinex'},
        n=input_len
        )

    # Perform work on the dataframe
    df = df_orig
    df = df.sort_index()
    df = df._get_numeric_data().drop(["period"], axis=1, errors='ignore')
    # filter out timestapm_ metrics
    df = df.filter(regex="(?!timestamp_.*)", axis=1)
    df = ta.add_all_ta_features(df, open="open", high="high", low="low",
                                close="close", volume="volume").dropna(axis=1)
    df_diff = (df - df.shift(1, fill_value=0)).\
        rename(lambda x: x+'_diff', axis=1)
    df = pd.concat([df, df_diff], axis=1)
    dataset = dw.normalize(df.values)
    target = df.columns.get_loc('close')
    y = dataset[:, target]
    history = {'loss': [], 'val_loss': []}

    # create the x and y train and val data
    x_train, y_train, x_val, y_val = dw.windowed(
        dataset, target, batch_size, history_size, step, lahead=0, ratio=1.0
        )

    preds = model.predict(x_train)

    print(preds)

    return preds, y_train, df

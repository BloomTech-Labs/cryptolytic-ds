from cryptolytic.start import init
import cryptolytic.model.model_framework as model_framework
import cryptolytic.model.data_work as dw
import cryptolytic.data.historical as h
import cryptolytic.data as d
import cryptolytic.model.model_framework as mfw

# tensorflow imports
import tensorflow as tf

# begin general external imports
import os
import ta
import pandas as pd
import time


params = {
        'history_size': 400,
        'lahead': 12*3,
        'step': 1,
        'period': 300,
        'batch_size': 200,
        'train_size': 10000
}

def run_model():
    # use default params
    init()
    history_size = 400
    input_len = 8000
    lahead = 12*3
    step = 2
    period = 300
    input_len = input_len + lahead
    now = int(time.time())
    start = now - input_len*period
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
    init()
    path = 'models/models/model_eth_btc.h5'
    model = tf.keras.models.load_model(path)

    # use default params
    now = time.time()
    now = int(now)
    params = {'filters1': 32, 'noise1': 0.007044669933974564,
              'filtershape1': [48, 48, 96], 'filtershape2': [64, 64, 128]}
    history_size = 400
    input_len = 8000
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
        n=input_len)

    # Perform work on the dataframe
    df = df_orig
    df = df.sort_index()
    df = df._get_numeric_data().drop(["period"], axis=1, errors='ignore')
    # filter out timestamp_ metrics
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

    return preds, y_train, df



def cron_pred2():
    """
    - Loads model for the given unique trading pair, gets the latest data 
    availble for that trading pair complete with 
    """
    init()
    all_preds = pd.DataFrame(columns=['close', 'api', 'trading_pair', 'exchange_id', 'timestamp'])

    for api, exchange_id, trading_pair in h.yield_unique_pair():
        model_path = mfw.get_model_path(api, exchange_id, trading_pair)
        if not os.path.exists(model_path):
            print(f'Model not available for {api}, {exchange_id}, {trading_pair}') 
            continue

        model = tf.keras.models.load_model(model_path)

        n = params['history_size']+params['lahead']

        df, dataset = h.get_latest_data(api,
                          exchange_id, trading_pair, 
                          params['period'], 
                          # Pull history_size + lahead length, shouldn't need more to make a 
                          # prediction
                          n=n)

        if df is None:
            continue

        target = df.columns.get_loc('close')

        print(dataset.shape)

        # Get the data in the same format that the model expects for its training
        x_train, y_train, x_val, y_val = dw.windowed(
            dataset, target,
            params['batch_size'], 
            params['history_size'],
            params['step'], 
            lahead=0, # Zero look ahead, don't truncate any data for the prediction
            ratio=1.0)


        if x_train.shape[0] < n:
            print(f'Invalid shape {x_train.shape[0]} in function cron_pred2')
            continue

        
        preds = model.predict(x_train)[:, 0][-params['lahead']]

        last_timestamp = df.timestamp[-1]
        timestamps = [last_timestamp + period * i for i in range(len(preds))]
        yo = pd.DataFrame({'close': preds, 'api': api, 'exchange': exchange_id, 'timestamp':  timestamps})
        all_preds = pd.concat([all_preds, yo], axis=1)

    return all_preds


def cron_train2():
    """
    - Loads model for the given unique trading pair, gets the latest data 
    availble for that trading pair complete with 
    """
    init()
    for api, exchange_id, trading_pair in h.yield_unique_pair():
        model_path = mfw.get_model_path(api, exchange_id, trading_pair)

        #model = tf.keras.load_model(path)

        n = params['train_size']

        df, dataset = h.get_latest_data(api,
                          exchange_id, trading_pair, 
                          params['period'], 
                          n=n)

        if df is None:
            continue

        target = df.columns.get_loc('close')

        print(n)
        print(df.shape, dataset.shape)

        # Get the data in the same format that the model expects for its training
        x_train, y_train, x_val, y_val = dw.windowed(
            dataset, 
            target,
            params['batch_size'], 
            params['history_size'],
            params['step'], 
            lahead=params['lahead'],
            ratio=0.8)

        print(x_train.shape)
        print(y_train.shape)
        print(x_val.shape)
        print(y_val.shape)
        if x_train.shape[0] < 10:
            print(f'Invalid shape {x_train.shape[0]} in function cron_train')
            continue


        model = mfw.create_model(x_train, params)


        print(x_train[0][0:5])

        # fit the model
        model = mfw.fit_model(model, x_train, y_train, x_val, y_val)
        model.save(model_path)

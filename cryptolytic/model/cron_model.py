# begin inter imports
from cryptolytic.start import init
import cryptolytic.model.model_framework as model_framework
import cryptolytic.model.data_work as dw
import cryptolytic.data.historical as h
import cryptolytic.data as d
import cryptolytic.model.model_framework as mfw
from cryptolytic import session
import cryptolytic.model.xgboost_model as xgmod
import cryptolytic.data.aws as aws
import cryptolytic.model.xgboost_model as xtrade
import cryptolytic.model.xgboost_arb_model as xarb
import pickle


# tensorflow imports
import tensorflow as tf

# begin general external imports
import os
import ta
import pandas as pd
import time
from io import StringIO
import gc



params = {
        'history_size': 400,
        'lahead': 12*3,
        'step': 1,
        'period': 300,
        'batch_size': 200,
        'train_size': 10000,
        'ncandles': 5000
}



def cron_pred():
    """
    - Loads model for the given unique trading pair, gets the latest data
    availble for that trading pair complete with
    """
    init()
    all_preds = pd.DataFrame(columns=['close', 'api', 'trading_pair', 'exchange_id', 'timestamp'])

    for exchange_id, trading_pair in h.yield_unique_pair(return_api=False):
        model_path = mfw.get_path('neural', model_path, exchange_id, trading_pair, '.h5')
        aws.download_file(model_path)
        if not os.path.exists(model_path):
            print(f'Model not available for {exchange_id}, {trading_pair}')
            continue

        model = tf.keras.models.load_model(model_path)

        n = params['history_size']+params['lahead']

        df, dataset = h.get_latest_data(
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


# TODO improve performance
def cron_train():
    """
    - Loads model for the given unique trading pair, gets the latest data
    availble for that trading pair complete with
    """
    init()

    now = int(time.time())
    pull_size = 5000

    h.live_update()

    for exchange_id, trading_pair in h.yield_unique_pair(return_api=False):
        # Loop until training on all the data want to train on or
        # if there is an error don't train
        start = int(now - params['ncandles'] * params['period'])
        # time_counter is used for batch processing
        time_counter = start
        while True:
            gc.collect()
            model_path = mfw.get_path('neural', exchange_id, trading_pair, '.h5')

            # model = tf.keras.load_model(path)

            n = params['train_size']

            # train in batches of 3000

            df, dataset = h.get_data(
                              exchange_id, trading_pair,
                              params['period'],
                              start=time_counter,
                              n=3000)

            if df is None:
                break

            print(df)

            time_counter = int(df.timestamp[-1])

            # finished training for this
            if time_counter >= now - params['period']:
                print('Finished training for  {api}, {exchange_id}, {trading_pair}')
                break

            if df.shape[0] < params['history_size']:
                break

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
                break

            # Create a model if not exists, else load model if it
            # not loaded
            model = None
            if not os.path.exists(model_path):
                model = mfw.create_model(x_train, params)
            # elif is for retraining for models
            elif model is None:
                model = tf.keras.models.load_model(model_path)

            # fit the model
            model = mfw.fit_model(model, x_train, y_train, x_val, y_val)
            print(f'Saved model {model_path}')
            model.save(model_path)
            # Upload file to s3
            aws.upload_file(model_path)

# be able to have model train on a large series of time without crashing
# split data into smaller batches


def xgb_cron_train(model_type):

    # Initialize the function and pull info from the .env file
    init()

    # Find the current time
    now = int(time.time())
    pull_size = 5000

    # Check for missing data, pull data from APIs if data is missing
    # h.live_update()

    # Check for every unqiue trading pair in each exchange
    for exchange_id, trading_pair in h.yield_unique_pair(return_api=False):
        # Loop until training on all the data want to train on or
        # if there is an error don't train
        start = int(now - params['ncandles'] * params['period'])
        time_counter = start

        gc.collect()
        model_path = mfw.get_path('models', model_type, exchange_id, trading_pair, '.pkl')

        n = params['train_size']

        # train in batches of 3000
        df, dataset = h.get_latest_data(
                            exchange_id,
                            trading_pair,
                            params['period'],
                            n=3000
                            )

        if df is None:
            break

        print(df)

        target = df.columns.get_loc('close')

        print(n)
        print(df.shape, dataset.shape)

        # Find the x and y train and test data
        x_train, y_train, x_test, y_test = xgmod.data_splice(df)

        # Create a model if not exists, else load model if it
        # not loaded
        model = None
        # TODO remove comment on below to restory functionality beyond testing enviornments
        # if not os.path.exists(model_path):
        if True:
            if model_type == 'trade':
                model = xgmod.create_model()

            elif model_type == 'arbitrage':
                # model = xgmod.create_model()
                pass
            # model = # Model training/update functions here

        # fit the model
        model = xgmod.fit_model(model, x_train, y_train)
        print(f'Saved model {model_path}')
        # model.save(model_path)
        pickle.dump(model, open(model_path, 'wb'))
        # Upload file to s3
        aws.upload_file(model_path)


# be able to have model train on a large series of time without crashing
# split data into smaller batches

def xgb_cron_pred(model_type='trade'):
    """
    - Loads model for the given unique trading pair, gets the latest data
    availble for that trading pair complete with
    """
    init()
    all_preds = pd.DataFrame(columns=['close', 'api', 'trading_pair', 'exchange_id', 'timestamp'])

    for exchange_id, trading_pair in h.yield_unique_pair(return_api=False):
        model_path = mfw.get_path('neural', model_type, exchange_id, trading_pair, '.pkl')
        if not os.path.exists(model_path):
            print(f'Model not available for {exchange_id}, {trading_pair}') 
            continue

        model = pickle.load(open(model_path, 'rb'))

        n = params['history_size']+params['lahead']

        df, dataset = h.get_latest_data(
                          exchange_id, trading_pair, 
                          params['period'], 
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
            lahead=0,  # Zero look ahead, don't truncate any data for the prediction
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

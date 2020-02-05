# begin inter imports
from cryptolytic.start import init
# init call early in imports so that cryptolytic.session imports correctly on
# Windows operating systems
init()
from cryptolytic.util import *
import cryptolytic.model.model_framework as model_framework
import cryptolytic.model.data_work as dw
import cryptolytic.data.historical as h
import cryptolytic.data as d
import cryptolytic.data.sql as sql
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

# TODO try maybe only use period from here
params = {
        'history_size': 400,
        'lahead': 12*3,
        'step': 1,
        'period': 300,
        'batch_size': 200,
        'train_size': 10000,
}

now = int(time.time())
_3months = 7890000
period = 300

Models = {
  'neural' : adict(
      # new models train from this date 
      # train data in batches, something can do with neural networks
      train_in_batches = True,
      history_size =  500,
      load_model_fn = tf.keras.models.load_model,
      save_model_fn = lambda model, model_path: model.save(model_path),
      step = 1,
      batch_size = 200,
      lahead = 12,
      Type='neural',
      train_size = 3000,
      prediction_data_size = 1500,
      feature_size = -1

  ),
  'trade' : adict(
    Type='trade',
    load_model_fn = lambda model_path: pickle.load(open(model_path, 'rb')),
    save_model_fn = lambda model, model_path: pickle.dump(model, open(model_path, 'wb')),
    prediction_data_size = 1500,
    lahead = 12,
    feature_size = 80,
    train_size = 10000,
  ),
'trade2' : adict(
    Type='trade',
    load_model_fn = lambda model_path: pickle.load(open(model_path, 'rb')),
    save_model_fn = lambda model, model_path: pickle.dump(model, open(model_path, 'wb')),
    prediction_data_size = 1500,
    lahead = 12,
    feature_size = 80,
    train_size = 10000,
  ),
  'arbitrage' : adict(
    Type='arbitrage',
    load_model_fn = lambda model_path: pickle.load(open(model_path, 'rb')),
    save_model_fn = lambda model, model_path: pickle.dump(model, open(model_path, 'wb')),
    prediction_data_size = 1500,
    lahead = 1,
    feature_size = 80,
    train_size = 10000,
  )
}

def get_latest_prediction(exchange_id, trading_pair, model):
    """Get timestamp for latest prediction"""
    q = """select "timestamp" from predictions 
           where trading_pair=%(trading_pair)s and exchange=%(exchange)s and 
                 model_type=%(model_type)s and period=%(period)s
           order by timestamp desc 
           limit 1;"""
    return sql.safe_q1(q, dict(trading_pair=trading_pair, exchange=exchange_id, 
                       model_type=model.Type, period=300))

def get_model_path(model, exchange_id, trading_pair):
    if model.Type == 'neural':
        return aws.get_path('models', model.Type, exchange_id, trading_pair, '.h5')
    elif model.Type == 'arbitrage' or model.Type == 'trade':
        return aws.get_path('models', model.Type, exchange_id, trading_pair, '.pkl')


# TODO get rid of train_size, add back start, don't have historical fail if there is little
# data available... maybe
# start should be based on the number of candles to pull
def train(model_name):
    init()
    Model = Models.get(model_name)
    assert Model is not None
    now = int(time.time())
    for exchange_id, trading_pair in h.yield_unique_pair(return_api=False):
        gc.collect()
        start = None
        train_in_batches = Model.get('train_in_batches')
        if train_in_batches==True:
            start = latest
        else:
            start = now - Model.train_size * period

        print('-'*20)
        print(exchange_id, trading_pair, start)

        latest = get_latest_prediction(exchange_id, trading_pair, Model)
    
        n = Model.train_size

        # Get the X and y data needed for training the model
        df, X, y = (h.get_data(exchange_id,
                       trading_pair,
                       start=start,
                       period=period,
                       Model=Model,
                       n=n))
Wow that's super interesting with Lathe. One of my favorite things about
Clojure is the runtime polymorphism it has along with functions like reify
and extend-type. core.matrix library for example has multiple java vector 
libraries you can use which would be like being able to swap out numpy 
with another library yourself even and it still have the same interface. 


        if df is None:
           print(f'dataframe was null {exchange_id} {trading_pair} {start} {n}')
           continue

        if Model.Type == 'trade' or Model.Type == 'arbitrage':
            # For training, the last points will not have a y labels
            # available for them (because they reference future values),
            # so truncatet he X set to be the same size as y
            X = X[:len(y)]

        if Model.Type == 'neural' and df.shape[0] < Model.get('history_size'):
            print(f'Available history size was too small to train {exchange_id} {trading_pair} {df.shape[0]}')
            continue

        print('df shape', df.shape)
        print('X shape', X.shape)
        print('y shape', y.shape)

        model = None
        model_path = get_model_path(Model, exchange_id, trading_pair)


        # Try loading the model if it has no predictions yet or it's set to train in batches
        if latest is not None and train_in_batches==True:
            try:
                aws.download_file(model_path)
                model = Model.load_model_fn(model_path)
            except Exception as e:
                print(f'Error {e}')
                print(f'Model not available for {exchange_id}, {trading_pair}')

        # Otherwise, create a new model
        # TODO put into function for model to avoid if statement
        else:
            if Model.Type == 'neural':
                model = mfw.create_model(X, Model)
            elif Model.Type == 'trade':
                model = xtrade.create_model()
            elif Model.Type == 'arbitrage':
                model = xarb.create_model()

        # Fit the model
        # TODO put into function for model to avoid if statement
        if Model.Type == 'neural':
            model = mfw.fit_model(model, X, y)
        elif Model.Type == 'trade':
            model = xtrade.fit_model(model, X, y)
        elif Model.Type == 'arbitrage':
            model = xarb.fit_model(model, X, y)

        # Save and the upload model
        Model.save_model_fn(model, model_path)
        aws.upload_file(model_path)


def pred(model_name):
    init()
    Model = Models.get(model_name)
    assert Model is not None
    now = int(time.time())
    for exchange_id, trading_pair in h.yield_unique_pair(return_api=False):
        gc.collect()
        # set the start time
        n = Model.prediction_data_size
        start = now - Model.train_size * period

        print(exchange_id, trading_pair, start)


        # Get the X and y data needed for training the model
        df, X, y = (h.get_data(exchange_id,
                       trading_pair,
                       period=period,
                       Model=Model,
                       start=start,
                       n=n))
        if df is None:
            print(f'dataframe was empty {exchange_id} {trading_pair} {start} {n}')
            continue
        elif df.shape[0] < Model.prediction_data_size:
            print(f'dataframe was small {df.shape} Expected at least {Model.prediction_data_size} {exchange_id} {trading_pair} {start} {n}')
            continue

        if Model.Type == 'neural' and df.shape[0] < Model.get('history_size'):
            print(f'Available history size was too small too predict on {exchange_id} {trading_pair} {df.shape[0]}')
            continue

        print('df shape', df.shape)
        print('X shape', X.shape)

        model = None
        model_path = get_model_path(Model, exchange_id, trading_pair)

        print(model_path)

        # Try loading the model if it has no predictions yet or it's set to train in batches
        try:
            aws.download_file(model_path)
            model = Model.load_model_fn(model_path)
        except Exception as e:
            print(f'Error {e}')
            print(f'Model not available for {exchange_id}, {trading_pair}')
            continue

        # Fit the model
        preds = None
        timestamps = None

        # TODO put into function for model to avoid if statement
        if Model.Type == 'neural':
            preds = model.predict(X)[:, 0][-Model.lahead:]
            timestamps = [df.timestamp[i] + Model.lahead * period for i in range(len(preds))]

        elif Model.Type == 'trade':
            preds = model.predict(X)
            timestamps = [df.timestamp[i] + Model.lahead * period for i in range(len(preds))]

        elif Model.Type == 'arbitrage':
            preds = model.predict(X)
            timestamps = [df.timestamp[i] + Model.lahead * period for i in range(len(preds))]

        # Insert predictions in database
        preds = pd.DataFrame(
            {'prediction': preds,
             'exchange': exchange_id,
             'timestamp':  timestamps,
             'trading_pair': trading_pair,
             'period': period,
             'model_type': Model.Type})

        sql.upsert(preds, 'predictions')

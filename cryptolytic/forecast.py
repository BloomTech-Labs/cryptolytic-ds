import tensorflow.keras.models as models
import cryptolytic.data as d
import os
import time
import numpy as np
from cryptolytic.util import *


def run():
    # Steps
    # 
    #̶ ̶L̶o̶a̶d̶ ̶s̶a̶v̶e̶d̶ ̶m̶o̶d̶e̶l̶
    # Grab latest data for all exchanges (start with binance)
    # run the model 
    # save forecast predictions to data folder

    # parameters
    history_size = 400
    input_len = 16000
    lahead = 12*3
    step = 2
    period = 300
    to_drop = lahead - 1
    input_len = input_len + to_drop
    batch_size = 200

    trading_pair = "eth_btc"
    exchange = "binance"
    period = 300
    n = batch_size  # predict using the last 1000 candles
    info = {"trading_pair": trading_pair,
            "exchange_id" : exchange,
            "period": period,
            "start": int(time.time()) - n*period}

    print(os.getcwd())
    
    model = models.load_model('models/filter/model_eth_btc.h5')
    df = d.get_df(info, n)
    df, dataset =  d.get_model_input_data(df)
    target = df.columns.get_loc('close')
    dataset = np.expand_dims(dataset, axis=0)

#    preds = model.predict(dataset)
#    return preds
    return dataset

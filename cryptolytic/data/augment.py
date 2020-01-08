import pandas as pd
from ta import add_all_ta_features

def resample_ohclv(df, time):
    ohlcv_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'}

    return df.resample(time, closed='left', label='left').apply(ohlcv_dict)

import ta
ta.add_momentum_ta
ta.add_volatility_ta
ta.add_volume_ta
ta.trend
# Daily Return (price change from t-1, t).
# result = ta.others.DailyReturnIndicator(df['close'], fillna=True).daily_return()
# Cumulative Return. ta.other.cumulative_return(close, fillna?)
# 

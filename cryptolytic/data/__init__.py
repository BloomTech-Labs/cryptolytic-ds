import numpy as np
import pandas as pd


def get_by_time(df, start, end):
    q = (df.index > start) & (df.index < end)
    return df[q]


def convert_datetime_to_timestamp(dt):
    """Convert pandas datetime to unix timestamp"""
    return int(dt.timestamp())


def denoise(signal, repeat):
    "repeat: how smooth to make the graph"
    copy_signal = np.copy(signal)
    for j in range(repeat):
        for i in range(3, len(signal)):
            # set previous timestep to be between the timestep i and i - 2
            copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
    return copy_signal


def merge_candle_dfs(df1, df2):
    """Merge candle dataframes"""
    merge_cols = ['trading_pair', 'exchange', 'period', 'datetime']
    df_merged = df1.merge(df2, how='inner', on=merge_cols) 
    return df_merged


def resample_ohlcv(df, period=None):
    """this function resamples ohlcv csvs for a specified candle interval; while
        this can be used to change the candle interval for the data, it can also be
        used to fill in gaps in the ohlcv data without changing the candle interval"""
    # dictionary specifying which columns to use for resampling
    ohlcv_dict = {'open': 'first',
                  'high': 'max',
                  'low': 'min',
                  'close': 'last',
                  'volume': 'sum'}

    # apply resampling
    if period==None:
        period = df['period'][0]
    period = pd.to_timedelta(period, unit='s')
    df_new = df.resample(period, how=ohlcv_dict)
    
    missing = nan_df(df_new)

    # df_new['trading_pair'] = df['trading_pair'][0]
    # df_new['exchange'] = df['exchange'][0]
    # df_new['exchange'] = df['exchange'][0]
    # df_new['period'] = df['period'][0]

    # return merge_candle_dfs(df, df_new)
    return missing


def nan_df(df):
    return df[df.isnull().any(axis=1)]


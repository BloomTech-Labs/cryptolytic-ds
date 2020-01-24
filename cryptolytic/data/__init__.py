import numpy as np
import pandas as pd
from cryptolytic.util import *
import cryptolytic.data.sql as sql


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
        for i in range(2, len(signal)):
            # set previous timestep to be between the timestep i and i - 2
            copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
    return copy_signal


# Normalize the dataset with (x-μ)/σ 
def normalize(df):
    pass


def merge_candle_dfs(df1, df2):
    """Merge candle dataframes"""
    merge_cols = ['trading_pair', 'exchange', 'period', 'datetime', 'timestamp']
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
    # likely to contain nils
    return df_new


def nan_df(df):
    return df[df.isnull().any(axis=1)]


def inner_merge(df1, df2):
    return df1.merge(df2, how='inner', on=(df1.columns & df2.columns).tolist())


def outer_merge(df1, df2):
    return df1.merge(df2, how='outer', on=(df1.columns & df2.columns).tolist())


def fix_df(df):
    """Changes columns to the right type if needed and makes sure the index is set as the
    datetime of the timestamp"""
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    numeric = ['period', 'open', 'close', 'high', 'low', 'volume', 'arb_diff', 'arb_signal']
    for col in numeric:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col])
    df = df.set_index('datetime')
    return df


def impute_df(df):
    """
    Finds the gaps in the time series data for the dataframe, and pulls the average market 
    price and its last volume for those values and places those values into the gaps. Any remaining
    gaps or new nan values are filled with backwards fill.
    """
    df = df.copy()
    gapped = resample_ohlcv(df) 
    gaps = nan_df(gapped).index
    # stop psycopg2 error with int conversion
    convert_datetime = compose(int, convert_datetime_to_timestamp)
    timestamps = mapl(convert_datetime, list(gaps)) 
    info = {'trading_pair': df['trading_pair'][0],
            'period': int(df['period'][0]),
            'exchange': df['exchange'][0],
            'timestamps': timestamps}
    if len(info['timestamps']) >= 2:
        avgs = sql.batch_avg_candles(info)
        volumes = sql.batch_last_volume_candles(info)
        df = outer_merge(df, avgs)
        df = outer_merge(df, volumes)

    df = fix_df(df)
    df['volume'] = df['volume'].ffill()
    df = df.bfill().ffill()
    assert df.isna().any().any() == False
    return df


def get_df(info, n=1000):
    """
    Pull info from database and give it some useful augmentation for analysis
    """
    df = sql.get_some_candles(info=info, n=n, verbose=True)
    df = impute_df(df)
    
    df['diff'] = df['high'] - df['low']
    df['diff2'] = df['close'] - df['open']
    dfarb = sql.get_arb_info(info, n)
    
    merged = merge_candle_dfs(df, dfarb)
    assert merged.isna().any().any() == False
    return merged

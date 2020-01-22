import numpy as np
import pandas as pd


def get_by_time(df, start, end):
    q = (df.index > start) & (df.index < end)
    return df[q]


def denoise(signal, repeat):
    "repeat: how smooth to make the graph"
    copy_signal = np.copy(signal)
    for j in range(repeat):
        for i in range(3, len(signal)):
            # set previous timestep to be between the timestep i and i - 2
            copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
    return copy_signal


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
    df = df.resample(period, how=ohlcv_dict)
    return df


def nan_rows(df):
    r = np.unique(np.where(df.isna())[0])

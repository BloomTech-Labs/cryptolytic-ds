import numpy as np
import ta
from scipy.stats import yeojohnson
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
    print("df", df)
    return df
    gapped = resample_ohlcv(df) 
    print("Gapped", gapped)
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
    
    df['high_m_low'] = df['high'] - df['low']
    df['close_m_open'] = df['close'] - df['open']
    dfarb = sql.get_arb_info(info, n)
    
    merged = merge_candle_dfs(df, dfarb)
    assert merged.isna().any().any() == False
    return merged


def thing(arg, axis=0):
    x = np.sign(arg) * np.log(np.abs(arg) + 1)
    mu = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    return x, mu, std


# Version 2 
def normalize(A):
    if isinstance(A, pd.DataFrame) or isinstance(A, pd.Series):
        A = A.values
    if np.ndim(A)==1:
        A = np.expand_dims(A, axis=1)
    A = A.copy()
    x, mu, std = thing(A, axis=0)
    for i in range(A.shape[1]):
        A[:, i] = (x[:, i] - mu[i]) / std[i]
    return A
   

def denormalize(values, df, col=None):
    values = values.copy()
    
    def eq(x, mu, std):
        return np.exp((x * std) + mu) - 1
    
    if np.ndim(values) == 1 and col is not None:
        x, mu, std = thing(df[col])
        return eq(values, mu, std)
    else:
        for i in range(values.shape[1]): 
            x, mu, std = thing(df.iloc[:, i])
            if isinstance(values, pd.DataFrame): 
                values.iloc[:, i] = eq(values.iloc[:, i], mu, std)
            else:
                values[:, i] = eq(values[:, i], mu, std)
        return values


def windowed(df, target, batch_size, history_size, step, lahead=1, ratio=0.8):
    xs = []
    ys = []
    
    x = df
    y = df[:, target]

    start = history_size # 1000
    end = df.shape[0] - lahead # 4990
    # 4990 - 1000 = 3990
    for i in range(start, end):
        indices = range(i-history_size, i, step)
        xs.append(x[indices])
        ys.append(y[i:i+lahead])
        
    xs = np.array(xs)
    ys = np.array(ys)
    
    nrows = xs.shape[0]
    train_size = int(nrows * ratio)
    # make sure the sizes are multiples of the batch size (needed for stateful lstm)
    train_size -= train_size % batch_size
    val_size = nrows - train_size
    val_size -= val_size  % batch_size
    total_size = train_size + val_size
    xs = xs[:total_size]
    ys = ys[:total_size]
    
    return xs[:train_size], ys[:train_size], xs[train_size:], ys[train_size:]


def get_model_input_data(df):
    """
    Will return a dataframe and a normalized version of that dataframe

    """
    df = df.copy()
    df = df.sort_index()
    df = df._get_numeric_data.drop(['period'], axis=1, errors='ignore')
    # maybe not necesarry, type of transform 
    df[['volume', 'high_m_low', 'arb_signal']].apply(lambda x: yeojohnson(np.float64(x))[0]).rename(lambda x: x+'_johnson', axis=1)
    df = df.filter(regex="(?!timestamp_.*)", axis=1) # filter out useless timestamp_ metrics
    df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume").dropna(axis=1)
    df_diff = (df - df.shift(1, fill_value=0)).rename(lambda x: x+'_diff', axis=1)
    df = pd.concat([df, df_diff], axis=1)
    dataset = normalize(df.values)
    target = df.columns.get_loc('close') 
    y = dataset[:, target]
    return df, dataset

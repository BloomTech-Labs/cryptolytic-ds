"""
Description: Functions for assessing data normality
"""
import pandas as pd
import numpy as np


def price_diff(df):
    """
    
    """
    pass

def fill_nan(df):
    """Iterates through a dataframe and fills NaNs with appropriate 
        open, high, low, close values."""

    # Forward fill close column.
    df['close'] = df['close'].ffill()
    # todo fill other columns as well

    return df

def find_outliers(s, window=30, sigma=10):
    """
    Find outliers in a series
    """
    avg = s.rolling(window=window).mean()
    residual = s - avg
    std = residual.rolling(window.window).std()
    outliers = (np.abs(residual) > std * sigma)
    return outliers

def get_outliers(df, dfs):
    """
    Return the time of abnormal data points (should be removed),
    which are points with an unexpected price or price change compared 
    to other exchanges.
    df: dataframe for a given trading pair (on a given exchange)
    dfs: dataframe for a given trading pair on all exchanges (might contain df)
    """

    pass



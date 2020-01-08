"""
Description: Functions for assessing data normality
"""
import pandas as pd
import numpy as np


def price_diff(df):
    """
    
    """

    pass

def singular_price(df):
    (0.02 * df['low']) + (0.02 * df['high']) + (.48 * df['open']) + (.48 * df['close'])


def get_outliers(df, dfs):
    """
    Return the time of abnormal data points (should be removed),
    which are points with an unexpected price or price change compared 
    to other exchanges.
    df: dataframe for a given trading pair (on a given exchange)
    dfs: dataframe for a given trading pair on all exchanges (might contain df)
    """

    pass



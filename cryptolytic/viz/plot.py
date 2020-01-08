import plotly.graph_objects as go 
import plotly
from cryptolytic.data import historical as h, sql
import logging

def candlestick(df):
    df2 = df
    if df.shape[0] > 10000:
        logging.warning('Not plotting more than 10000 candles')
        df2 = df[0:10000]
    fig = go.Figure(data=[go.Candlestick(
                            x=df2.index,
                            open=df2['open'],
                            close=df2['close'],
                            high=df2['high'],
                            low=df2['low'])])
    return fig

def plot_all_candlesticks(df):
    for api, exchange_id, trading_pair in h.yield_unique_pair():
        print(api, exchange_id, trading_pair)
        q = ((df['exchange'] == exchange_id) & (df['api'] == api) & (df['trading_pair']==trading_pair))
        dfq = df[q]
        if dfq.shape[0] <= 0: 
            continue
        yield candlestick(dfq)

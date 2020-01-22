import plotly.graph_objects as go 
import plotly
from cryptolytic.data import historical as h, sql
import logging

def candlestick(df):
    if df.shape[0] > 10000:
        logging.warning('Not plotting more than 10000 candles')
        df = df[0:10000]
    fig = go.Figure(data=[go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            close=df['close'],
                            high=df['high'],
                            low=df['low'])])
    return fig

def plot_all_candlesticks(df):
    for groupkey, group  in df.groupby(['api', 'exchange', 'trading_pair']):
        api, exchange_id, trading_pair = groupkey
        print(api, exchange_id, trading_pair)
        yield candlestick(group)

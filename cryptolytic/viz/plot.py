import plotly.graph_objects as go 
import plotly
from cryptolytic.data import historical as h, sql

def candlestick(df):
    fig = go.Figure(data=[go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            close=df['close'],
                            high=df['high'],
                            low=df['low'])])
    return fig

def plot_all_candlesticks(df):
    for api, exchange_id, trading_pair in h.yield_unique_pair():
        print(api, exchange_id, trading_pair)
        q = ((df['exchange'] == exchange_id) & (df['api'] == api) & (df['trading_pair']==trading_pair))
        dfq = df[q]
        if dfq.shape[0] <= 0: 
            continue
        yield candlestick(dfq)

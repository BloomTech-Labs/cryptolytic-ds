import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly
from cryptolytic.data import historical as h, sql
import logging

def candlestick(df):
    """Expects a dataframe with open, high, low, close data. Plots candlesticks using plotly."""
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
    """
    Will plot a candlestick chart for each api, exchange, trading_pair, and period
    """
    for groupkey, group  in df.groupby(['api', 'exchange', 'trading_pair', 'period']):
        api, exchange_id, trading_pair = groupkey
        print(api, exchange_id, trading_pair)
        yield candlestick(group)


def model_prediction(plot_data, future, title):
    """
    Using plot data 
    """
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
#    time_steps = create_time_steps(plot_data.shape[0])

    plt.title(title)
    for i, x in enumerate(plot_data):
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                    label=labels[i])

    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

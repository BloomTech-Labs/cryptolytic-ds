import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly
import numpy as np
from matplotlib.pylab import rcParams
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

def model_performance(df):
    pass

# def model_prediction(, future, title):
#     """
#     Using plot data 
#     """
#     labels = ['History', 'True Future', 'Model Prediction']
#     marker = ['.-', 'rx', 'go']
# #    time_steps = create_time_steps(plot_data.shape[0])
# 
#     plt.title(title)
#     for i, x in enumerate(plot_data):
#         plt.plot(future, plot_data[i], marker[i], markersize=10,
#                     label=labels[i])
# 
#     plt.legend()
#     plt.xlim(plot_data.index, (future+5)*2])
#     plt.xlabel('Time-Step')
#     return plt


def plot_arbitration(df):
    """
    Used to visualize arbitration on a single exchange
    """
    fig, ax1 = plt.subplots()
    plt.title('Arbitration Graph')
    
    color = 'tab:red'
    ax1.set_ylabel('Data', color=color)
    ax1.plot(df.index, df.close, color=color, label=f'{df.exchange[0]}')
    ax1.tick_params(axis='y', labelcolor=color)
    color ='tab:green'
    ax1.plot(df.index, df.avg, color=color, label='Mean Closing Price')
    ax1.legend(loc='lower left')
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Signal', color=color)  
    ax2.plot(df.index,d.denoise(df.arb_signal, 20), color=color, label='Signal')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='lower right')


def simple_plot(df, x):
    rcParams['figure.figsize'] = 20,4
    plt.plot(df.index, np.log(df['close']), color='pink')
    plt.plot(df.index, np.log(df['close'].rolling(x).mean()), color='black')


def plot_states(ts_vals, states, time_vals):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_ylabel('Data',         color=color)
    ax1.plot(time_vals, ts_vals,      color=color)
    ax1.tick_params(axis='y',            labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Hidden state', color=color)  
    ax2.plot(time_vals,states,     color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'{len(np.unique(states))} State Model')
    fig.tight_layout()  
    plt.show()

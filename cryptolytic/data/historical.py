"""
    Description: Contains functions on APIs and turning that into candlestick data.
    Fully Implemented: Cryptowatch, Poloniex, CoinAPI

"""
import requests
from cryptolytic.util import date
import time
import os
import requests
import json
import datetime
import numpy as np
import pandas as pd

# Json conversion dictionary for cryptocurrency abbreviations
crypto_name_table = None
with open('data/cryptocurrencies.json', 'r') as f:
    crypto_name_table = json.load(f)
assert crypto_name_table.keys()

api_info = None
with open('data/api_info.json', 'r') as f:
    api_info = json.load(f)
assert len(api_info) > 1

# TODO consider putting into a more general file
def crypto_full_name(crypto_short):
    """crypto_short: e.g. eth, btc
       result: e.g. Ethereum, Bitcoin"""
    #print(crypto_name_table)
    lookup = crypto_name_table.get(crypto_short.upper())
    return lookup

def trading_pair_info(api, x):
    """Returns full info for the trading pair necessary for the exchange.
    x: btc_eth or something. Probably should be a vector actually like ["eth" "btc"] instead.
    """
    # btcheth style trading pairs
    baseId, quoteId = x.split('_')
    handled = False

    if api in {'cryptowatch', 'hitbtc', 'bitfinex'}:
        x = x.replace('_', '')
        handled = True
    if api in {'poloniex', 'hitbtc', 'bitfinex'}:
        x = x.upper()
        handled = True
    if api in {'coincap'}:
        baseId = crypto_full_name(baseId).lower()
        quoteId = crypto_full_name(quoteId).lower()
        handled = True
    if not handled:
        raise Exception('API not supported ', api)

    return {'baseId' : baseId,
            'quoteId' : quoteId,
            'trading_pair' : x}

def convert_candlestick(candlestick, api, timestamp):
    # dict with keys :open, :high, :low, :close, :volume, :quoteVolume, :timestamp
    candlestick = candlestick.copy()
    candlestick['timestamp'] = timestamp

    if api=='poloniex':
        pass
    elif api=='coincap':
        pass
#        candlestick['quoteVolume'] = -1 # Unknown quote volume
    elif api=='cryptowatch':
        candlestick = {
            'close' : candlestick[0],
            'open'  : candlestick[1],
            'high'  : candlestick[2],
            'low'   : candlestick[3],
            'volume' : candlestick[4],
#            'quoteVolume' : candlestick[5]
        }
    elif api=='bitfinex':
        candlestick = {
            'open'  : candlestick[1],
            'close' : candlestick[2],
            'high'  : candlestick[3],
            'low'   : candlestick[4],
            'volume' : candlestick[5],
#            'quoteVolume' : -1 # unknownquoteVolume
        }
    elif api=='hitbtc':
        # timestamp is another format assume the default timestamp is correct instead
        candlestick['high'] = candlestick.pop('max')
        candlestick['low'] = candlestick.pop('min')
#       candlestick['quoteVolume'] = candlestick.pop('volumeQuote')
    else:
        raise Exception('API not supported ', api)

    return candlestick

# Exchange, Trading Pair, Api Key, Period, After
# Probably better to not to do start calls
# TODO
# take
# coincap intervals: m1, m5, m15, m30, h1, h2, h4, h8, h12, d1, w1
# associate them with num

def format_apiurl(api, params={}):
    url = None
    params = params.copy()
    # Coincap expects milliseconds in its url query
    if api in {'coincap'}:
        params['start'] *= 1000
        params['end'] *= 1000
    # Standard URL query
    if api in {'cryptowatch', 'poloniex', 'coincap', 'hitbtc', 'bitfinex'}:
        url = api_calls[api].format(**params)
    # Hasn't been verified
    else:
        raise Exception('API not supported', api)
    print(url)
    return url

# Coincap uses time intervals like h1, m15 etc. so need a function to convert to the seconds to that format
def get_time_interval(api, period):
    """For coincap, hitbtc, etc. which use intervals"""
    minutes = period / 60
    hours = minutes / 60
    days = hours / 24
    weeks = days / 7

    accepted_values = {
        'coincap' : {
            'weeks' : [1],
            'days' : [1],
            'hours' : [1, 2, 4, 8, 12],
            'minutes' : [1, 5, 15, 30],
        },
        'hitbtc' : {
            'days' : [1, 7],
            'hours' : [1, 4],
            'minutes' : [1, 3, 5, 15, 30]
        },
        'bitfinex' : {
            'minutes' : [1, 5, 15, 30],
            'hours' : [1, 3, 6, 12],
            'days' : [1, 7, 14],
        }
    }.get(api)

    if accepted_values == None:
        raise Exception('API not supported', api)

    elif weeks >= 1 and accepted_values.has('weeks'):
        x = list(filter(lambda x: x<hours, accepted_values['weeks']))
        interval = 'w'+str(np.max(x))
    elif days >= 1:
        x = list(filter(lambda x: x<hours, accepted_values['days']))
        interval = 'd'+str(np.max(x))
    elif hours >= 1:
        x = list(filter(lambda x: x<hours, accepted_values['hours']))
        interval = 'h'+str(np.max(x))
    else:
        x = [1] + list(filter(lambda x: x<minutes, accepted_values['minutes']))
        interval = 'm'+str(np.max(x))

    # expects uppercase like 'M1' for 1 minute
    if api in {'hitbtc'}:
        interval = interval.upper()
    if api in {'bitfinex'}: # expect 1m format instead
        print('oetnsuh')
        interval = interval[1:] + interval[0]

    return interval

def conform_json_response(api, json_response):
    """Get the right data from the json response. Expects a list, either like [[],...], or like [{},..]"""
    if api=='cryptowatch':
        return json_response['result'][str(period)]
    elif api=='coincap':
        return json_response['data']
    elif api in {'poloniex', 'hitbtc', 'bitfinex'}:
        return json_response
    else:
        raise Exception('API not supported', api)
    return None

def get_from_api(api='cryptowatch', exchange='binance', trading_pair='btceth',
                 period=14400, interval=None, apikey=None):
    """period : candlestick length in seconds. Default 4 hour period.
       interval : time interval, either unix or %d-%m-%Y format
    """
    # Check some things
    assert exchange in exchanges

    if api in {'cryptowatch'} and apikey==None:
        raise Exception('Missing API key')

    # Variable initialization
    pair_info = trading_pair_info(api, trading_pair)
    baseId = pair_info.get('baseId') # the first coin in the pair
    quoteId = pair_info.get('quoteId') # the second coin in the pair
    trading_pair = pair_info.get('trading_pair') # e.g. eth_usd in the form of what the api expects
    start = date.convert_datetime(interval[0]) # start time unix timestamp
    end = date.convert_datetime(interval[1])  # end time unix timestamp
    limit=100 # 100 limit by default, should change depending on the API later.
    assert start < end

    urlparams = dict(exchange=exchange, trading_pair=trading_pair, apikey=apikey,
                     period=period, end=end, baseId=baseId, quoteId=quoteId,
                     limit=limit)

    # Uses another notation for period, fix if needed
    if api in {'coincap', 'hitbtc', 'bitfinex'}:
        urlparams['interval'] = get_time_interval(api, period)


    # Get all the candle sticks in the time period.
    current_timestamp = start
    candles = []
    while current_timestamp < end:
        urlparams['start']=current_timestamp
        url = format_apiurl(api, urlparams)

        # TODO more error checking and rescheduling if there is an error and the current_timestamp has been updated at least one (i.e. > start)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"""In function get_from_api, got bad response {response.status_code}. Exiting early.""")
            break

        json_response = json.loads(response.content)
        candlestick_info = conform_json_response(api, json_response)
        assert len(candlestick_info) > 0
        # Add to candles list.
        for i, candle in enumerate(candlestick_info):
            current_timestamp += period
            candlenew = convert_candlestick(candle, api, current_timestamp)
            candles.append(candlenew)

        # Sleep for a second
        time.sleep(5)

        break # TODO remove

    # Check if candle schema is valid
    candle_schema = ['timestamp', 'open', 'close', 'volume', 'high', 'low', 'quoteVolume']
    assert all(x in candles[0].keys() for x in candle_schema)


    # TODO reschedule task and log error if the last_timestamp is not >= end
    # TODO drop duplicates
    return dict(
         api=api,
         exchange = exchange,
         candles=candles,
         start  = start,
         end    = end,
         last_timestamp    = current_timestamp,
         candles_collected = len(candles),
         period = str(datetime.timedelta(seconds=period)))

def collect_data():
    get_from_api(api=api,
                exchange=exchange,
                trading_pair = trading_pair,
                period=period,
                interval=[start, end],
                apikey=os.getenv(apikey))
    
def live_update():
    for api, api_data in api_info.items():
        api_exchanges = api_data['exchanges']
        for exchange_id, exchange_data in api_exchanges.items():
            print(api)
            print(api_exchanges)
            print(exchange_data['trading_pairs'])

def test_get_candles():
    start = '01-01-2019'
    start = date.convert_datetime(start)
    end = start + 14400*2

    candle_info = get_from_api(api='bitfinex',
                            exchange='binance',
                            trading_pair ='eth_btc',
                            period=14400,
                            interval=[start, end],
                            apikey=os.getenv('_cryptowatch_private_key'))

    assert candle_info['candles'][0].keys()

    return candle_info

#    get_candles(
#        interval=['01-12-2011', '01-12-2018'],
#        api='cryptowatch',
#        trading_pairs=['bch_usd'])
# "https://api.cryptowat.ch/markets/poloniex/ethbtc"

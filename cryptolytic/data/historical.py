"""
    Description: Contains functions on APIs and turning that into candlestick data.
"""
import requests
from cryptolytic.util import date
from cryptolytic.data import sql
import time
import os
import requests
import json
from datetime import datetime
import numpy as np
import pandas as pd
from collections import deque
import logging
import ciso8601


# Json conversion dictionary for cryptocurrency abbreviations
crypto_name_table = None
with open('data/cryptocurrencies.json', 'r', encoding='utf-8') as f:
    crypto_name_table = json.load(f)
assert crypto_name_table.keys()


"""
If you are having timeout issues connecting to the AWS RDS instance, make sure
to configure your AWS VPC security groups to allow outside access
"""
api_info = None
with open('data/api_info.json', 'r') as f:
    api_info = json.load(f)
assert len(api_info) > 1


def crypto_full_name(crypto_short):
    """crypto_short: e.g. eth, btc
       result: e.g. Ethereum, Bitcoin"""
    # print(crypto_name_table)
    lookup = crypto_name_table.get(crypto_short.upper())
    return lookup


def trading_pair_info(api, x):
    """Returns full info for the trading pair necessary for the exchange.
    x: e.g. btc_eth
    """
    # btcheth style trading pairs
    baseId, quoteId = x.split('_')
    handled = False

    if api_info.get(api).get("pair_no_underscore"):
        x = x.replace('_', '')
        handled = True
    if api_info.get(api).get("pair_uppercase"):
        x = x.upper()
        handled = True
    if api_info.get(api).get("pair_dash_seperator"):
        x = x.replace('_', '-')
        handled = True
    if api in {'coincap'}:
        baseId = crypto_full_name(baseId).lower()
        quoteId = crypto_full_name(quoteId).lower()
        handled = True
    if not handled:
        raise Exception('API not supported ', api)

    return {'baseId'      : baseId,
            'quoteId'     : quoteId,
            'trading_pair': x}


def convert_candlestick(candlestick, api, timestamp):
    # dict with keys :open, :high, :low, :close, :volume, :quoteVolume, :timestamp
    candlestick_old = candlestick
    candlestick = candlestick.copy()
    ohclv = ["open", "high", "close", "low", "volume", "timestamp"]
    corder = api_info[api].get('candlestick_order')

    if api_info[api].get('candlestick_no_conversion'):
        pass
    # reorder candlestick information
    elif corder is not None:
        candlestick = {k: candlestick[i] for k, i in 
                       zip(ohclv, corder)}
    elif api=='hitbtc':
        candlestick['high'] = candlestick.pop('max')
        candlestick['low'] = candlestick.pop('min')
    elif api=='coincap':
        candlestick['timestamp'] = candlestick.pop('period')
    else:
        raise Exception('API not supported ', api)

    timestamp_format = api_info.get(api).get("candle_timestamp_format")
    try:
        if timestamp_format == "milliseconds":
            candlestick['timestamp'] = candlestick['timestamp'] // 1000
        elif timestamp_format == "iso8601":
            candlestick['timestamp'] = int(ciso8601.parse_datetime(candlestick['timestamp']).timestamp())
        elif timestamp == "replace":
            candlestick['timestamp'] = timestamp

        # Check if candle schema is valid

        # no less than year 2000 or greater than 2030. timesamp must be an int
        if (not all(x in candlestick.keys() for x in ohclv)) or \
           not isinstance(candlestick['timestamp'], int) or candlestick['timestamp'] >= 1894131876 or candlestick['timestamp'] <= 947362914: 
            raise Exception()
    except Exception:
        raise Exception("API: ", api, "\nInvalid Candle: ", candlestick, "\nOld candle: ", candlestick_old)

#    candlestick['timestamp'] = timestamp
    # only keep these keys, otherwise can mess up insertion
    return {key: candlestick[key] for key in ohclv}


def format_apiurl(api, params={}):
    """
        Format the url to get the candle data for the given api
    """
    url = None
    params = params.copy()
    # Coincap expects milliseconds in its url query
    if api_info.get(api).get("timestamp_format") == "milliseconds":
        params['start'] *= 1000
        params['end']  *= 1000
    if api_info.get(api).get("timestamp_format") == "iso8601":
        params['start'] = datetime.utcfromtimestamp(params['start'])
        params['end'] = datetime.utcfromtimestamp(params['end'])
    # Standard URL query
    if api_info.get(api).get("timestamp_format") == "seconds":
        pass
    elif api_info.get(api) is None:
        # Hasn't been verified
        raise Exception('API not supported', api)
    url = api_info[api]['api_call'].format(**params)
    print(url)
    return url


# Coincap uses time intervals like h1, m15 etc. so need a function to convert to the seconds to that format
def get_time_interval(api, period):
    """For coincap, hitbtc, etc. which use a format like 'm1' instead of
       period like 60 for 60 seconds."""
    minutes = period / 60
    hours = minutes / 60
    days = hours / 24
    weeks = days / 7

    accepted_values = api_info.get(api).get("time_interval")

    if accepted_values is None:
        raise Exception('API not supported', api)
    elif weeks >= 1 and accepted_values.has('weeks'):
        x = list(filter(lambda x: x<=weeks, accepted_values['weeks']))
        interval = 'w'+str(np.max(x))
    elif days >= 1:
        x = list(filter(lambda x: x<=days, accepted_values['days']))
        interval = 'd'+str(np.max(x))
    elif hours >= 1:
        x = list(filter(lambda x: x<=hours, accepted_values['hours']))
        interval = 'h'+str(np.max(x))
    else:
        x = [1] + list(filter(lambda x: x<=minutes, accepted_values['minutes']))
        interval = 'm'+str(np.max(x))

    # expects uppercase like 'M1' for 1 minute
    if api_info.get(api).get("uppercase_timeinterval"):
        interval = interval.upper()
    elif api_info.get(api).get("reverse_timeinterval"):  # expect 1m format instead
        interval = interval[1:] + interval[0]

    return interval


def conform_json_response(api, json_response):
    """Get the right data from the json response. Expects a list, either like [[],...], or like [{},..]"""
    if api=='cryptowatch':
        return json_response['result'][str(period)]  # TODO fix
    elif api=='coincap':
        return json_response['data']
    elif api in {'poloniex', 'hitbtc', 'bitfinex', 'coinbase'}:
        return json_response
    else:
        raise Exception('API not supported', api, 'Response was ', json_response)
    return None


def get_from_api(api='cryptowatch', exchange='binance', trading_pair='eth_btc',
                 start=1546300800, limit=100, period=300, apikey=None):
    """period: candlestick length in seconds. Default 300 seconds.
       limit: number of candles to pull
    """
    if api in {'cryptowatch'} and apikey==None:
        raise Exception('Missing API key')

    # Variable initialization
    pair_info = trading_pair_info(api, trading_pair)
    baseId = pair_info.get('baseId')  # the first coin in the pair
    quoteId = pair_info.get('quoteId')  # the second coin in the pair
    trading_pair_api = pair_info.get('trading_pair')  # e.g. eth_usd in the form of what the api expects
    start = start  # start time unix timestamp
    end = start+period*limit  # end time unix timestamp
    cutoff_time = int(time.time()-(period / 2))  # don't ask for a time greater than -period / 2 seconds ago
    end = min(end, cutoff_time)
    assert start < end

    print("Start", start)
    print("End", end)


    # parameters for the url to get candle data from
    urlparams = dict(exchange=exchange, trading_pair=trading_pair_api, apikey=apikey,
                     period=period, end=start+(limit*period), baseId=baseId, quoteId=quoteId,
                     limit=limit)

    # The API uses another notation for period (like 1m for 1 minute)
    if api in {'coincap', 'hitbtc', 'bitfinex'}:
        urlparams['interval'] = get_time_interval(api, period)

    urlparams['start']=start
    urlparams['end']=end
    url = format_apiurl(api, urlparams)

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"In function get_from_api, got bad response {response.status_code}. Exiting early.",
                        f"Response Content: {response.content}")

    # load and convert the candlestick info to be a common format
    json_response = json.loads(response.content)
    candlestick_info = conform_json_response(api, json_response)

    assert len(candlestick_info) > 0

    # Get all the candle sticks in the time period.
    candles = []
    current_timestamp = start
    for candle in candlestick_info:
        candlenew = convert_candlestick(candle, api, current_timestamp)
        current_timestamp = candlenew['timestamp']
        candles.append(candlenew)

    # return the candlestick information
    return dict(
        api=api,
        exchange=exchange,
        candles=candles,
        last_timestamp=current_timestamp,
        trading_pair=trading_pair,
        candles_collected=len(candles),
        period=period)  # period in seconds


def yield_unique_pair():
    """Yield unique trading pair (not including period information)"""
    api_iter = api_info.items()
    for api, api_data in api_iter:
        api_exchanges = api_data['exchanges']
        for exchange_id, exchange_data in api_exchanges.items():
            for trading_pair in exchange_data['trading_pairs']:
                yield (api, exchange_id, trading_pair)


def sample_every_pair(n=3000, query={}):
    """Function that tries to sample for every pair that matches the query
        into a dataframe.
       query: paremeters for the db query"""
    df = pd.DataFrame()
    # can't specify these with this function
    assert not {'api', 'exchange_id', 'trading_pair'}.issubset(query)
    assert {'period'}.issubset(query)

    def dict_matches(cond, b):
        return set(cond.items()).issubset(set(b.items()))

    def select_keys(d, keys):
        return {k: d[k] for k in keys if k in d}

    def filter_pairs():
        keys = ['api', 'exchange_id', 'trading_pair']
        thing = select_keys(query, keys)
        return filter(lambda pair: 
                      dict_matches(thing, dict(zip(keys, pair))), 
                      yield_unique_pair())

    for api, exchange_id, trading_pair in filter_pairs():
        d = {'api': api,
             'exchange_id': exchange_id,
             'trading_pair': trading_pair}
        d.update(query)  # mutates
        df = df.append(sql.get_some_candles(d, n, verbose=True))
    return df


# returns true if updated, or None if the task should be dropped
def update_pair(api, exchange_id, trading_pair, timestamp, period=300, num_retries=0):
    if num_retries > 100:
        return

    # limit to 100 candles if limit is not specified
    limit = api_info.get(api).get('limit') or 100 
    candle_info = None

    try:  # Get candle information
        candle_info = get_from_api(api=api,
                                   exchange=exchange_id,
                                   trading_pair=trading_pair,
                                   start=timestamp,
                                   period=period,
                                   limit=limit)
    except Exception as e:
        print(e)
        logging.error(e)

    # If the last timestep is equal to the ending candle
    # that there is such a gap in candle data that the time frame cannot
    # advance to new candles, so continue with this task at an updated timestep
    if candle_info is None or candle_info['last_timestamp'] == timestamp:
        print(type(limit))
        print(type(timestamp))
        print(type(num_retries))
        print(type(period))
        return update_pair(api, exchange_id, trading_pair, timestamp+limit*period, period, num_retries + 1)
    
    # Print the timestamp
    ts = candle_info['last_timestamp']
    print(datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    # Insert into sql
    try:
        print("Adding Candlestick to database", api, exchange_id, trading_pair, timestamp)
        sql.candlestick_to_sql(candle_info)
        return True # ran without error
    except Exception as e:
        logging.error(e)


def live_update(period = 300): # Period default is 5 minutes
    """
        Updates the database based on the info in data/api_info.json with new candlestick info,
        grabbing data from the last timestamp until now, with the start date set at the start of 2019.
    """
    now = time.time()

    # use a deque to rotate the tasks, and pop them when they are done. 
    # this is to avoid sending too many requests to one api at once.
    d = deque()
    
    for api, exchange_id, trading_pair in yield_unique_pair():
        d.append([api, exchange_id, trading_pair])
    
    for i in range(10_000):
        if len(d)==0:
            break
        api, exchange_id, trading_pair = d[-1]  # get the current task
        print(api, exchange_id, trading_pair)

        start = sql.get_latest_date(exchange_id, trading_pair, period) or 1546300800  # timestamp is January 1st 2019
            
        # already at the latest date, remove 
        if start >= round(now)-period:
            d.pop() 
            continue

        # returns true if updated, or None if the task should be dropped
        result = update_pair(api, exchange_id, trading_pair, start, period)
        if result is None:
            d.pop()
            continue
        else:
            d.rotate()


# hitbtc
def fill_missing_candles():
    missing = sql.get_missing_timesteps()
    print(missing)
    for i, s in missing.iterrows():
        api, exchange, period, trading_pair, timestamp, ntimestamp = s
        print(int(timestamp))
        # first try to update with an api call
        update_pair(api, exchange, trading_pair, int(timestamp), int(period))

    # For those that are still missing, impute their value
    missing = sql.get_missing_timesteps()
    for i, s in missing.iterrows():
        api, exchange, period, trading_pair, timestamp, ntimestamp = s
        candles = []
        last_timestep = 0
        for ts in range(timestamp, ntimestamp, int(period)):
            print("Imputing values for ", exchange, trading_pair, ts)
            candle = sql.get_avg_candle({'timestamp': ts,
                                         'trading_pair': trading_pair,
                                         'period': period,
                                         'exchange': exchange})
            last_timestep = candle['timestamp'] = ts
            candles.append(candle)

        candle_info = dict(
            api=api,
            exchange=exchange,
            candles=candles,
            last_timestamp=last_timestep,
            trading_pair=trading_pair,
            candles_collected=len(candles),
            period=period)  # period in seconds

        sql.candlestick_to_sql(candle_info)

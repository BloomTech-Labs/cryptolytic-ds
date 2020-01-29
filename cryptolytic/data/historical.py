"""
    Description: Contains functions on APIs and turning that into candlestick data.
"""
import requests
from cryptolytic.util import date
from cryptolytic.util import *
from cryptolytic.data import sql
import cryptolytic.model.data_work as dw
import cryptolytic.data as d
import ta
import time
import os
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

api_info = {k: v for k, v in api_info.items() if not v.get('disabled')}


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

    if api_info.get(api).get("pair_reverse"):
        temp = baseId
        baseId = quoteId
        quoteId = temp
        x = baseId + '_' + quoteId
        handled = True
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
    elif api=='poloniex':
        candlestick['timestamp'] = candlestick.pop('date')
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
    if api_info[api].get("timestamp_format") == "milliseconds":
        params['start'] *= 1000
        params['end']  *= 1000
    if api_info[api].get("timestamp_format") == "iso8601":
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
        return list(json_response['result'].values())[0]
    elif api=='coincap':
        return json_response['data']
    elif api in {'poloniex', 'hitbtc', 'bitfinex', 'coinbase'}:
        return json_response
    else:
        raise Exception('API not supported', api, 'Response was ', json_response)
    return None

def lookup_apikey(api):
    apikey = api_info[api].get('apikey')
    if apikey is not None:
        return os.environ.get(apikey)


def get_from_api(api='cryptowatch', exchange='binance', trading_pair='eth_btc',
                 start=1546300800, limit=100, period=300):
    """period: candlestick length in seconds. Default 300 seconds.
       limit: number of candles to pull.
       start: start time in unix timestamp format
    """

    # 
    apikey = lookup_apikey(api)

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


def update_pair(api, exchange_id, trading_pair, timestamp, period=300,
                num_retries=0):
    """Returns true if updated, or None if the task should be dropped"""
    # exit if num retries > 100
    if num_retries > 20:
        return

    now = time.time()
    now = int(now)

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
        print(f'Error encountered: {e}')
        logging.error(e)

    # If the last timestep is equal to the ending candle
    # that there is such a gap in candle data that the time frame cannot
    # advance to new candles, so continue with this task at an updated timestep
    if candle_info is None or candle_info['last_timestamp'] == timestamp:
        if timestamp >= now - 86400:  # seconds in a day
            return
        print(f'Retry {api} {exchange_id} {trading_pair} {timestamp} {num_retries}')
        return update_pair(api, exchange_id, trading_pair, timestamp+limit*period, period, num_retries + 1)

    # Print the timestamp
    ts = candle_info['last_timestamp']
    print(datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    # Insert into sql
    try:
        print("Adding Candlestick to database", api, exchange_id, trading_pair, timestamp)

        sql.candlestick_to_sql(candle_info)
        return True  # ran without error
    except AssertionError as e:
        logging.error(e)


def live_update(period=300):  # Period default is 5 minutes
    """
        Updates the database based on the info in data/api_info.json with
        new candlestick info, grabbing data from the last timestamp until now,
        with the start date set at the start of 2019.
    """
    now = time.time()
    now = int(now)

    # use a deque to rotate the tasks, and pop them when they are done.
    # this is to avoid sending too many requests to one api at once.
    d = deque()

    for api, exchange_id, trading_pair in yield_unique_pair():
        d.append([api, exchange_id, trading_pair])

    for i in range(10_000):
        if len(d) == 0:
            break
        api, exchange_id, trading_pair = d[-1]  # get the current task

        print(api, exchange_id, trading_pair)

        start = sql.get_latest_date(exchange_id, trading_pair, period) or 1546300800 # timestamp is January 1st 2019

        # already at the latest date, remove
        if start >= now-86400:
            d.pop()
            continue

        # Returns true if updated, or None if the task should be dropped
        result = update_pair(api, exchange_id, trading_pair, start, period)
        if result is None:
            d.pop()
            continue
        else:
            d.rotate()


def fill_missing_candles():
    missing = sql.get_missing_timesteps()

    for i, s in missing.iterrows():
        api, exchange, period, trading_pair, timestamp, ntimestamp = s
        print(int(timestamp))
        # first try to update with an api call
        update_pair(api, exchange, trading_pair, int(timestamp), int(period))


def get_data(api, exchange_id, trading_pair, period, start, n=8000):
    """
    Get data for the given trading pair and perform feature engineering on that data
    """

    print(mapl(lambda x: type(x), [api, exchange_id, trading_pair, period, start, n]))
    df_orig = d.get_df({'start': start, 'period': period, 'trading_pair': trading_pair,
              'exchange_id': exchange_id}, n=n)
    df = df_orig

    # # some times a small of candles will be returned and it won't work wi
    # if df.shape[0] < 1:
    #     return

    try:
        df = df.sort_index()
        df = df._get_numeric_data().drop(["period"], axis=1, errors='ignore')
        # filter out timestamp_ metrics
        df = df.filter(regex="(?!timestamp_.*)", axis=1)
        df = ta.add_all_ta_features(df, open="open", high="high", low="low",
                                    close="close", volume="volume").fillna(axis=1, value=0)
        df_diff = (df - df.shift(1, fill_value=0)).rename(lambda x: x+'_diff', axis=1)
        df = pd.concat([df, df_diff], axis=1)
        dataset = np.nan_to_num(dw.normalize(df.values), nan=0)

        return df, dataset
    except Exception as e:
        # Returns None None for tuple unpacking convineance
        return None, None
        print('Warning: Error in get_data: {e}')


def get_latest_data(api, exchange_id, trading_pair, period, n=8000):
    """
    Get data for the given trading pair and perform feature engineering on that data for the latest date
    """
    now = int(time.time())
    start = now - n*period

    return get_data(api, exchange_id, trading_pair, period, start,  n=n)

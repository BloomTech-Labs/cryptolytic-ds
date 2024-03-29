import psycopg2 as ps
import os
import cryptolytic.data as d
from cryptolytic.data import historical
from cryptolytic.util import *
import cryptolytic.util.date as date
import time
import pandas as pd
import json
from itertools import repeat


ohlc = ["open", "high", "low", "close"]

def get_credentials():
    """Get the credentials for a psycopg2.connect"""
    return {
        'dbname': os.getenv('POSTGRES_DBNAME'),
        'user': os.getenv('POSTGRES_USERNAME'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'host': os.getenv('POSTGRES_ADDRESS'),
        'port': int(os.getenv('POSTGRES_PORT'))
    }


def get_conn():
    conn = ps.connect(**get_credentials())
    cursor = conn.cursor()
    return conn, cursor


def safe_q(q,  args={}, return_conn=False):
    """Safe sql query"""
    conn, cur = get_conn()
    try:
        cur.execute(q, args)
        if return_conn:
            return conn, cur
        else:
            return cur
    except ps.OperationalError as e:
        sql_error(e)
        return


def safe_q1(q, args={}, return_conn=False):
    result = safe_q(q, args, return_conn).fetchone()
    if result is not None:
        return result[0]

# like q1 but more appropriate in some cases
def safe_q2(q, args={}, return_conn=False):
    result = safe_q(q, args, return_conn).fetchone()
    return result


def safe_qall(q, args={}, return_conn=False):
    return safe_q(q, args, return_conn).fetchall()


def sql_error(error):
    """
        Documentation: http://initd.org/psycopg/docs/errors.html
    """
    # TODO put into log or something
    print("SQL Error:")
    print(f"Error Code: {error.pgcode}\n")


def check_tables():
    return safe_qall("""SELECT * FROM pg_catalog.pg_table
                      WHERE schemaname != 'pg_catalog'
                      AND schemaname != 'information_schema';""")


def create_candle_table():
    q = """CREATE TABLE candlesticks
                (api text not null,
                 exchange text not null,
                 trading_pair text not null,
                 timestamp bigint not null,
                 period numeric not null,
                 open numeric not null,
                 close numeric not null,
                 high numeric not null,
                 low numeric not null,
                 volume numeric not null,
                 primary key (exchange, trading_pair, timestamp, period)
                 );"""

    # imputed boolean not null default FALSE
    conn, cur = safe_q(q, return_conn=True)
    if conn is not None:
        conn.commit()


def drop_candle_table():
    conn, cur = safe_q("DROP TABLE IF EXISTS candlesticks;", return_conn=True)
    if conn is not None:
        conn.commit()


def get_table_schema(table_name):
    q = """select column_name, data_type, character_maximum_length
           from INFORMATION_SCHEMA.COLUMNS where table_name = %(table_name)s"""
    return safe_qall(q, {'table_name': table_name})


def get_table_columns(table_name):
    q = """
        select column_name
        from INFORMATION_SCHEMA.COLUMNS where table_name = %(table_name)s;"""
    results = safe_qall(q, {'table_name': table_name})
    return list(map(lambda x: x[0], results))


def add_data_to_table(df, cur=None, table='candlesticks'):
    """Builds a string from our data-set using the mogrify method which is
        then called once using the execute method to insert the candlestick 
        information (collected using functions in the historical file), into the 
        database. 
    """

    order = get_table_columns(table)
    n = len(order)
    query = "("+",".join(repeat("%s", n))+")"
    df = d.fix_df(df)
    
    print(df.head())
    print(len(df))
    args_str = None

    conn = None

    if cur is None:
        conn = ps.connect(**get_credentials())
        cur = conn.cursor()

    try:
        x = [
            str(
                cur.mogrify(query, row), encoding='utf-8'
                ) for row in df[order].values]

        args_str = ','.join(x)
    except Exception as e:
        print('ERROR', e)
    try:
        cur.execute(f"INSERT INTO {table} VALUES" + args_str + " on conflict do nothing;")
        if conn is not None:
            conn.commit()
    except ps.OperationalError as e:
        sql_error(e)
        return


def candlestick_to_sql(data):
    """
        Inserts candlesticks data into database. See get_from_api in data/historical.py for more info.
    """

    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    dfdata = pd.concat(
            [pd.DataFrame(data['candles']), pd.DataFrame(data)], axis=1
            ).drop(
                    ['candles', 'candles_collected', 'last_timestamp'],
                    axis=1)
    add_data_to_table(dfdata, cur)
    conn.commit()


def get_latest_date(exchange_id, trading_pair, period):
    """
        Return the latest date for a given trading pair on a given exchange
    """
    q = """
        SELECT timestamp FROM candlesticks
        WHERE exchange=%(exchange_id)s AND trading_pair=%(trading_pair)s AND period=%(period)s
        ORDER BY timestamp desc
        LIMIT 1;
    """
    latest_date = safe_q1(q, {'exchange_id': exchange_id,
        'trading_pair': trading_pair,
        'period': period
        })
    if latest_date is None:
        print('No latest date')

    return latest_date


def get_earliest_date(exchange_id, trading_pair, period):
    """
        Return the earliest date for a given trading pair on a given exchange
    """
    q = """
        SELECT timestamp FROM candlesticks
        WHERE exchange=%(exchange_id)s AND trading_pair=%(trading_pair)s AND period=%(period)s
        ORDER BY timestamp asc
        LIMIT 1;
    """
    latest_date = safe_q1(q, {'exchange_id': exchange_id,
        'trading_pair': trading_pair,
        'period': period})
    if latest_date is None:
        print('No latest date')

    return latest_date


def get_some_candles(info, n=10000, verbose=False):
    """
        Return n candles
        info: can contain start (unix-timestamp or str), end, exchange_id, 
            period (in seconds), trading_pair
            Example: info={'start':1546300800, 'end':1546309800, 'exchange_id':'bitfinex',
                           'trading_pair':'eth_btc', 'period':300}
    """
    n = min(n, 50000)  # no number larger than 50_000
    select = "open, close, high, low, timestamp, volume" if not verbose else "*"
    where = ''
	
    if 'period' not in info.keys():
        info['period'] = 300

    # make sure dates are of right format
    if 'start' in info:
        info['start'] = date.convert_datetime(info['start'])
    if 'end' in info:
        info['end'] = date.convert_datetime(info['end']) 

    def add_clause(where, key, clause):
        if key in info.keys():
            if len(where) == 0:
                where = "WHERE " + clause + " "
            else:
                where += "AND " + clause + " "
        return where

    where = add_clause(where, 'exchange_id', "exchange=%(exchange_id)s")
    where = add_clause(where, 'start', "timestamp >= %(start)s")
    where = add_clause(where, 'end', "timestamp <= %(end)s")
    where = add_clause(where, 'period', "period = %(period)s")
    where = add_clause(where, 'trading_pair', "trading_pair=%(trading_pair)s")

    q = f"""
        SELECT {select} FROM candlesticks
        {where}
        ORDER BY timestamp asc
        LIMIT {n};
        """
    results = safe_qall(q, info)
    columns = get_table_columns('candlesticks') if select == "*" else ["open", "close", "high", "low", "timestamp", "volume"]
    # TODO instead of returning a dataframe, return the query and then either convert to a dataframe (with get_candles) or to json
    df = pd.DataFrame(results, columns=columns)  
    df['period'] = info['period']
    return d.fix_df(df)

def get_api(api):
    q = "SELECT * FROM candlesticks WHERE api = %(api)s"
    safe_qall(q, {'api': api})


def get_bad_timestamps(info):
    q = """
    select * from (select "timestamp", lead("timestamp", 1) over (order by "timestamp") ntimestamp
    from candlesticks
    where exchange=%(exchange_id)s and trading_pair=%(trading_pair)s and "period"=%(period)s q
    where "timestamp" <> ntimestamp - 60;"""
    assert {''}.issubset(info.keys())
    return safe_qall(q, info)


def remove_duplicates():
    """ 
        Remove any duplicate candlestick information from the database. 
    """
    q = """
        with q as (select *, "timestamp" - lag(timestamp, 1)
                over (partition by(exchange, trading_pair, period) 
                order by "timestamp"
        ) as diff from candlesticks)
        delete from candlesticks
        where ctid in (
                select ctid 
                from q
                where diff=0
                order by timestamp);
            """

    conn, curr = safe_q(q, return_conn=True)
    if conn is not None:
        conn.commit()


def get_missing_timesteps():
    q = """
        select api, exchange, period, trading_pair, "timestamp",  "timestamp" + diff as ntimestamp
        from (select *, lead(timestamp, 1)
        over (partition by(exchange, trading_pair, period) order by "timestamp") - "timestamp" as diff
        from candlesticks) q
        where diff <> "period";
    """
    missing = safe_qall(q)

    return pd.DataFrame(missing, columns = ["api", "exchange", "period", "trading_pair", "timestamp", "ntimestamp"])



# Used for filling in missing candlestick values
# expects timestamp, trading_pair, and period
def get_avg_candle(query):
    """Query to get avg price values for a candlestick at a certain timestamp.
       TODO batch query for improved performance."""

    assert {'timestamp', 'trading_pair', 'period', 'exchange'}.issubset(query.keys())

    q =  """select avg("open"), avg(high), avg(low), avg("close")  from candlesticks
            where "timestamp"=%(timestamp)s and trading_pair=%(trading_pair)s and period=%(period)s;"""
    intermediate = safe_qall(q, query)

    # Query to get previous volume for the trading pair
    q2 = """select prev_volume from (select *, lag(volume, 1) over
            (partition by (exchange, trading_pair, period)
            order by "timestamp") as prev_volume from candlesticks) q
            where trading_pair=%(trading_pair)s and exchange=%(exchange)s and period=%(period)s and timestamp=%(timestamp)s
;    """
    ohlc = ["open", "high", "low", "close", "timestamp"]
    result = {key: intermediate[i] for i, key in zip(range(len(intermediate)), ohlc)}
    result['volume'] = safe_q1(q2, query)

    return result


def batch_avg_candles(info):
    assert {'timestamps', 'trading_pair', 'period', 'exchange'}.issubset(info.keys())

    assert len(info['timestamps']) >= 2
    info['timestamps'] = tuple(info['timestamps'])

    q = """
        select "timestamp", avg("open"), avg(high), avg(low), avg("close") from candlesticks
        where timestamp in %(timestamps)s and trading_pair=%(trading_pair)s and period=%(period)s
        group by timestamp;
    """

    result = safe_qall(q, info)
    df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df[ohlc] = df[ohlc].apply(pd.to_numeric)
    return d.fix_df(df)


def batch_last_volume_candles(info):
    assert {'timestamps', 'trading_pair', 'period', 'exchange'}.issubset(info.keys())
    info_copy = info.copy()
    # get the previous volumes, so minus the period, forward 
    # fills if this still has nans
    info['timestamps'] = mapl(lambda x: x-info['period'], info['timestamps'])

    assert len(info['timestamps']) >= 2
    info['timestamps'] = tuple(info['timestamps'])

    q = """with sub as (select "timestamp", volume from candlesticks
        where trading_pair=%(trading_pair)s and "period"=%(period)s and
        "timestamp" in %(timestamps)s and exchange=%(exchange)s)
        select "timestamp"+%(period)s, volume from sub;"""

    volumes = safe_qall(q, info)

    volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
    df = pd.DataFrame({'timestamp': info_copy['timestamps']})
    df = df.merge(volumes, how='left', on='timestamp')
    return d.fix_df(df.ffill().bfill())
    



def get_arb_info(info, n=1000):
    """
    Example: info := {'start':1556668800, 'period':300, 'trading_pair':'eth_btc', 'exchange_id':'binance'}
    """

    assert {'exchange_id', 'trading_pair', 'period', 'start'}.issubset(info.keys())
    info['n'] = n

    q = """with sub as (
           select * from candlesticks
           where trading_pair=%(trading_pair)s and period=%(period)s and timestamp>=%(start)s
           ),

           thing as (
           select "timestamp", avg(close) from  sub
           group by (timestamp)
           )

          select exchange,trading_pair, thing.timestamp, "period", "avg", "close"-"avg" as arb_diff, ("close"-"avg")/"avg" as arb_signal from
                 (sub inner join thing on sub.timestamp = thing.timestamp)
          where exchange=%(exchange_id)s
          order by thing.timestamp
          limit %(n)s;
    """

    results = safe_qall(q, info)
    if results is not None:
        # arb_signal is more interpretable than arb_diff but the signal is the same
        df = pd.DataFrame(results, columns=["exchange", "trading_pair", "timestamp", "period", "avg", "arb_diff", "arb_signal"])
        return d.fix_df(df)


def create_predictions_table():
    q = """CREATE TABLE predictions
                (exchange text not null,
                 model_type text not null,
                 trading_pair text not null,
                 timestamp bigint not null,
                 period numeric not null,
                 prediction text not null,
                 primary key (model_type, exchange, trading_pair, timestamp, period)
                 );"""

    # imputed boolean not null default FALSE
    conn, cur = safe_q(q, return_conn=True)
    if conn is not None:
        conn.commit()

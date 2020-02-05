import psycopg2 as ps
import os
import cryptolytic.data as d
from cryptolytic.data import historical
from cryptolytic.util import *
import cryptolytic.util.date as date
import time
import pandas as pd
import json
import numpy as np
from itertools import repeat
from psycopg2.extensions import register_adapter, AsIs
ps.extensions.register_adapter(np.int64, ps._psycopg.AsIs)


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
    """Connect to postgres database, return connection and cursor"""
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


def safe_q1(q, args={}):
    """Safe q but return 1 result"""
    result = safe_q(q, args, False).fetchone()
    if result is not None:
        return result[0]


# like q1 but more appropriate in some cases
def safe_q2(q, args={}):
    result = safe_q(q, args, False).fetchone()
    return result


def safe_qall(q, args={}, return_conn=False):
    return safe_q(q, args, False).fetchall()


def sql_error(error):
    """Documentation: http://initd.org/psycopg/docs/errors.html"""
    print("SQL Error:")
    print(f"Error Code: {error.pgcode}\n")


def check_tables():
    """Query to show tables in database"""
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
                 primary key (exchange, trading_pair, timestamp, period));"""

    conn, cur = safe_q(q, return_conn=True)
    if conn is not None:
        conn.commit()


def get_table_schema(table_name):
    q = """select column_name, data_type, character_maximum_length
           from INFORMATION_SCHEMA.COLUMNS where table_name = %(table_name)s"""
    return safe_qall(q, {'table_name': table_name})


def get_table_columns(table_name, primary_only=False):
    """Get the columns (or only the columns which are primary_keys with primary_only option) of the given table."""
    if primary_only:
        q = """
            SELECT c.column_name, c.data_type
            FROM information_schema.table_constraints tc 
            JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name) 
            JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
               AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
            WHERE constraint_type = 'PRIMARY KEY' and tc.table_name = %(table_name)s;"""
    else:
        q = """
            select column_name
            from INFORMATION_SCHEMA.COLUMNS where table_name = %(table_name)s;"""

    results = safe_qall(q, {'table_name': table_name})
    return list(map(lambda x: x[0], results))


def mogrified(df, table_name=None):
    """Serialize values in a format that is fast for insertion into the given table."""

    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    order = None
    if table_name is not None:
        order = get_table_columns(table_name)
    else:   
        order = df.columns

    n = len(order)
    query = "("+",".join(repeat("%s", n))+")"
    try: 
        x = [
            str(
                cur.mogrify(query, row), encoding='utf-8'
                ) for row in df[order].values]

        args_str = ','.join(x)
        return args_str
    except Exception as e:
        print(f'Error {e} oeua')

# build a where clause conveniently
def add_clause(where, clause):
    if len(where) == 0:
        where = "WHERE " + clause + " "
    else:
        where += "AND " + clause + " "
    return where


def upsert(df):
    """Update with insert, creates a temporary table then uses that table to update, maybe not the best way."""

    primary_columns = get_table_columns(table_name, primary_only=True)
    columns = get_table_columns(table_name)
    mogged = mogrified(df, table_name)
    temp_table = "temporary_table"
    where_clause = ""
    view_table = f"{table_name}_view"

    for column in primary_columns:
        where_clause = add_clause(where_clause, f"{view_table}.{column} = {temp_table}.{column}")


    new_columns = str(tuple(mapl(lambda x: f"{temp_table}."+x, columns))).replace("'", '')
    columns = str(tuple(columns)).replace("'", '"')

    # Update table by having a temporary table hold the values
    q = f"""
    create temp table if not exists {temp_table} as 
        select * from {table_name} limit 0;
    insert into {temp_table} values """ + mogged + ";" + \
    f"""
    create or replace view {view_table} as select * from {table_name};
    update {view_table}
    set {columns} = {new_columns}
    from {temp_table}
    {where_clause};
    drop table {temp_table};
    """

    conn, cur = safe_q(q, {}, return_conn=True)
    if conn is not None:
        conn.commit()

    # Did update for existing data, now insert any new data
    q = f"insert into {table_name} values " + mogged + " on conflict do nothing;"
    conn, cur = safe_q(q, {}, return_conn=True)
    if conn is not None:
        conn.commit()


def candlestick_to_sql(data):
    """Inserts candlesticks data into database. See get_from_api in data/historical.py for more info."""

    df = pd.concat(
            [pd.DataFrame(data['candles']), pd.DataFrame(data)], axis=1
            ).drop(
                    ['candles', 'candles_collected', 'last_timestamp'],
                    axis=1)

    print(df.head())
    upsert(df, "candlesticks")


def get_latest_date(exchange_id, trading_pair, period):
    """Return the latest date for a given trading pair on a given exchange"""
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
    """Return the earliest date for a given trading pair on a given exchange"""
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
    obt = 'order by "timestamp" desc'

    if 'period' not in info.keys():
        info['period'] = 300

    # make sure dates are of right format
    if 'start' in info:
        info['start'] = date.convert_datetime(info['start'])
        where = add_clause(where, "timestamp >= %(start)s")
        # if start is supplied, don't order by timestamp descending
        obt = 'order by "timestamp" asc'
        
    if 'end' in info:
        info['end'] = date.convert_datetime(info['end']) 

    if 'end' in info: where = add_clause(where, "timestamp <= %(end)s")
    if 'period' in info: where = add_clause(where, "period = %(period)s")
    if 'trading_pair' in info: where = add_clause(where, "trading_pair=%(trading_pair)s")
    if 'exchange_id' in info: where = add_clause(where, "exchange=%(exchange_id)s")

    # If start is not supplied, will pull the latest n candles
    q = f"""
        create or replace view thing2 as (
            select {select} from candlesticks
            {where}
            {obt}
            limit {n}
        );
        select * from thing2
        order by "timestamp" asc;
        """
    results = safe_qall(q, info)
    safe_q("drop view if exists thing2;")
    columns = get_table_columns('candlesticks') if select == "*" else ["open", "close", "high", "low", "timestamp", "volume"]
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


def batch_avg_candles(info):
    assert {'timestamps', 'trading_pair', 'period', 'exchange'}.issubset(info.keys())

    info = info.copy()
    assert len(info['timestamps']) >= 2
    timestamp_df = pd.DataFrame({'timestamps': info['timestamps']})

    info['timestamps'] = mogrified(timestamp_df)


    q = f"""
        drop table if exists thing2;

        create or replace view something2 as(
	    select * from candlesticks c
        	where c.trading_pair = %(trading_pair)s  and c.period = %(period)s
        );
        create temp table thing2 as (
                select * from 
                (values {info['timestamps']}) as t (blah)
                inner join something2 s on s.timestamp = t.blah);
                        
        select "timestamp", avg("open"), avg(high), avg(low), avg("close") from thing2
        group by "timestamp";
        """

    result = safe_qall(q, info)
    df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df[ohlc] = df[ohlc].apply(pd.to_numeric)
    return d.fix_df(df).sort_index()


# TODO redo to use the same logic as above
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
    

def get_avg_close(info):
    assert {'start', 'end', 'period', 'trading_pair'}.issubset(info)
    q = """
        select "timestamp", avg(close) from candlesticks
        group by ("timestamp", "period", trading_pair)
        having "timestamp" >= %(start)s and "timestamp" <= %(end)s and period=%(period)s and 
            trading_pair=%(trading_pair)s;
    """
    df = pd.DataFrame(safe_qall(q, info), columns=['timestamp', 'avg'])
    df['avg'] = df['avg'].astype(float)
    return df


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

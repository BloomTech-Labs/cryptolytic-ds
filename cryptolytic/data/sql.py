import psycopg2 as ps
import os
from cryptolytic.data import historical
import time
import pandas as pd
import json
from itertools import repeat


def get_credentials():
    """Get the credentials for a psycopg2.connect"""
    return {
        'dbname': os.getenv('POSTGRES_DBNAME'),
        'user': os.getenv('POSTGRES_USERNAME'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'host': os.getenv('POSTGRES_ADDRESS'),
        'port': int(os.getenv('POSTGRES_PORT'))
    }


def sql_error(error):
    """
        Documentation: http://initd.org/psycopg/docs/errors.html
    """
    # TODO put into log or something
    print("SQL Error:")
    print(f"Error Code: {error.pgcode}\n")


def check_tables():
    creds = get_credentials()
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    query = """SELECT * FROM pg_catalog.pg_tables
               WHERE schemaname != 'pg_catalog'
               AND schemaname != 'information_schema';"""
    try:
        cur.execute(query)
    except ps.OperationalError as e:
        sql_error(e)
        return
    results = cur.fetchall()
    print(results)


def create_candle_table():
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    query = """CREATE TABLE candlesticks
                (api text not null,
                 exchange text not null,
                 trading_pair text not null,
                 timestamp bigint not null,
                 period numeric not null,
                 open numeric not null,
                 close numeric not null,
                 high numeric not null,
                 low numeric not null,
                 volume numeric not null);"""
    try:
        cur.execute(query)
    except ps.OperationalError as e:
        sql_error(e)
        return
    conn.commit()


def drop_candle_table():
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    query = """DROP TABLE IF EXISTS candlesticks;"""
    try:
        cur.execute(query)
    except ps.OperationalError as e:
        sql_error(e)
        return
    conn.commit()


def check_candle_table():
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    print('Checking Table candlesticks')
    query = """SELECT * FROM candlesticks"""
    cur.execute(query)
    results = cur.fetchall()
    print(results)


def get_table_schema(table_name):
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    query = f"""
        select column_name, data_type, character_maximum_length
        from INFORMATION_SCHEMA.COLUMNS where table_name = '{table_name}';
    """
    try:
        cur.execute(query)
        return cur.fetchall()
    except ps.OperationalError as e:
        sql_error(e)
        return


def get_table_columns(table_name):
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    query = f"""
        select column_name
        from INFORMATION_SCHEMA.COLUMNS where table_name = '{table_name}';
    """
    try:
        cur.execute(query)
        return list(map(lambda x: x[0], cur.fetchall()))
    except ps.OperationalError as e:
        sql_error(e)
        return


def add_candle_data_to_table(df, cur):
    """
        Builds a string from our data-set using the mogrify method which is
        then called once using the execute method
    """
 
    order = get_table_columns('candlesticks')
    n = len(order)
    query = "("+",".join(repeat("%s", n))+")"
    df['timestamp'] = df['timestamp'].apply(str)

    try:
        x = [
            str(
                cur.mogrify(query, row), encoding='utf-8'
                ) for row in df[order].values]

        args_str = ','.join(x)
    except Exception as e:
        print('ERROR', e)
        
    try:
        cur.execute("INSERT INTO candlesticks VALUES" + args_str)
    except ps.OperationalError as e:
        sql_error(e)
        return

def get_latest_date(exchange_id, trading_pair):
    """
        Return the latest date for a given trading pair on a given exchange
    """
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    query = """
        SELECT timestamp FROM candlesticks
        WHERE exchange=%(exchange_id)s AND trading_pair=%(trading_pair)s
        ORDER BY timestamp desc
        LIMIT 1;
    """
    latest_date = None
    try:
        cur.execute(query,
                    {'exchange_id': exchange_id,
                     'trading_pair': trading_pair})
        latest_date = cur.fetchone()
        if latest_date is not None:
            return latest_date[0]
        print('No latest date')
    except ps.OperationalError as e:
        sql_error(e)
        return
    return latest_date

def get_some_candles(info, n=100):
    """
        Return n candles
    """
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    n = min(n, 10000) # no number larger than 10_000
    query = f"""
        SELECT open, close, high, low, timestamp, volume FROM candlesticks
        WHERE exchange=%(exchange_id)s and trading_pair=%(trading_pair)s and 
              period=%(period)s and timestamp >= %(start)s and timestamp <= %(end)s
        LIMIT {n} ;
    """
    try:
        cur.execute(query, info)
        results = cur.fetchall()
        # TODO probably 
        df = pd.DataFrame(results, columns=get_table_columns('candlesticks'))
        return df
    except ps.OperationalError as e:
        sql_error(e)
        return


def candlestick_to_sql(data):
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    dfdata = pd.concat(
        [pd.DataFrame(data['candles']), pd.DataFrame(data)], axis=1
                       ).drop(
                           ['candles', 'candles_collected', 'last_timestamp'],
                           axis=1)
    add_candle_data_to_table(dfdata, cur)
    conn.commit()

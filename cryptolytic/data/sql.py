import psycopg2 as ps
import os
from cryptolytic.data import historical
import time
import pandas as pd
import json

def get_credentials():
    """Get the credentials for a psycopg2.connect"""
    return {
        'dbname' : os.getenv('POSTGRES_DBNAME'),
        'user': os.getenv('POSTGRES_USERNAME'),
        'password' : os.getenv('POSTGRES_PASSWORD'),
        'host' : os.getenv('POSTGRES_ADDRESS'),
        'port' : int(os.getenv('POSTGRES_PORT'))
    }
def sql_error(error):
    """
        Documentation: http://initd.org/psycopg/docs/errors.html
    """
    # TODO put into log or something
    print("SQL Error:")
    print(f"Error Code: {e.pgcode}\n")

def check_tables():
    print("hey")
    creds = get_credentials()
    print(creds)
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    print("hey2")
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
                (api text primary key not null,
                 exchange text not null,
                 trading_pair text not null,
                 timestamp bigint not null,
                 open numeric not null,
                 close numeric not null,
                 high numeric not null,
                 low numeric not null,
                 volume bigint not null);"""
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


def add_candle_data_to_table(dfdata=[], trading_pair='etc_btc'):

    # open connection to the AWS RDS
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    
    query = """
        INSERT INTO candlesticks(api, exchange, trading_pair, timestamp, open, close, high, low, volume)
        VALUES (%(api)s, %(exchange)s, %(trading_pair)s, %(timestamp)s, %(open)s, %(close)s, %(high)s, %(low)s, %(volume)s);
    """
    try:
        cur.execute(
        query,
        {
            'api': dfdata['api'],
            'exchange': dfdata['exchange'],
            'trading_pair': str(trading_pair),
            'timestamp': dfdata['timestamp'],
            'open': dfdata['open'],
            'close': dfdata['close'],
            'high': dfdata['high'],
            'low': dfdata['low'],
            'volume': dfdata['volume']
        }
    )
    except ps.OperationalError as e:
        sql_error(e)
        return
      
    conn.commit()


def get_latest_date(exchange_id, trading_pair):
    """
        Return the latest date for a given trading pair on a given exchange
    """
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    query = """
        SELECT * FROM candlesticks
        WHERE exchange={exchange_id} AND trading_pair={trading_pair}
        ORDER BY timestamp desc
        LIMIT 1;
    """
    latest_date = None
    try:
        cur.execute(query)
        latest_date = cur.fetchone()
    except ps.OperationalError as e:
        sql_error(e)
        return
    return latest_date or 1546300800 

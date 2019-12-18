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
                 volume numeric not null);"""
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


def add_candle_data_to_table():

    # open connection to the AWS RDS
    conn = ps.connect(**get_credentials())
    cur = conn.cursor()
    data = historical.get_from_api(api = 'bitfinex', exchange='bitfinex', trading_pair='eth_btc', interval=[(time.time()-100000), time.time()])

    dfdata = pd.concat([pd.DataFrame(data['candles']), pd.DataFrame(data)], axis=1).drop(['candles', 'candles_collected', 'last_timestamp', 'start', 'end', 'period'], axis=1)


    for index, row in dfdata.iterrows():
        query = """INSERT INTO candlesticks(
            close,
            high,
            low,
            open,
            timestamp,
            volume,
            api,
            exchange
        )
                """ + str(row['close']) + ' ' + str(row['high']) + ' ' + str(row['low']) + ' ' + str(row['open']) + ' ' + str(row['timestamp']) + ' ' + str(row['volume']) + ' ' + str(row['api']) + ' ' + str(row['exchange']) + ';'
        # execute and commit the query
        cur.execute(query)
    conn.commit()

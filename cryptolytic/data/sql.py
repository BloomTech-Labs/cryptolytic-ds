import psycopg2 as ps
import os
from cryptolytic.data import historical
import cryptolytic.util.date as date
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
                 volume numeric not null);"""

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


def get_latest_date(exchange_id, trading_pair):
    """
        Return the latest date for a given trading pair on a given exchange
    """
    q = """
        SELECT timestamp FROM candlesticks
        WHERE exchange=%(exchange_id)s AND trading_pair=%(trading_pair)s
        ORDER BY timestamp desc
        LIMIT 1;
    """
    latest_date = safe_q1(q, {'exchange_id': exchange_id,
                              'trading_pair': trading_pair})
    if latest_date is None:
        print('No latest date')

    return latest_date

def get_some_candles(info, n=100, verbose=False):
    """
        Return n candles
    """
    n = min(n, 50000) # no number larger than 50_000
    select = "open, close, high, low, timestamp, volume" if not verbose else "*"
    where = ''
    
    # make sure dates are of right format
    if 'start' in info:
        info['start'] = date.convert_datetime(info['start'])
    if 'end' in info:
        info['end'] = date.convert_datetime(info['end']) 

#    # start is supplied but end is not
#    if 'start' in info and 'end' not in info:
#        period = info.get('period') or 300
#        info['end'] = info['start'] + n * period

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
    return df


def get_candles(info, n=100, verbose=False):
    df = get_some_candles(info, n, verbose)
    numeric = ['period', 'open', 'close', 'high', 'low', 'volume']
    df[numeric] = df[numeric].apply(pd.to_numeric)
    return df

def get_api(api):
    q = "SELECT * FROM candlesticks WHERE api = %(api)s"
    safe_qall(q, {'api': api})


def remove_api(api):
    """
    Drop API from candle table
    """
    q = """DELETE FROM candlesticks
           WHERE api = %(api)s"""
    conn, cur = safe_q(q, {'api' : api}, return_conn=True)
    if conn is not None:
        print(f"Removed {api}")
        conn.commit()

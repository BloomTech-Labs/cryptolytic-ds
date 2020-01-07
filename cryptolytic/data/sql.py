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


def get_conn():
    conn = ps.connect(**sql.get_credentials())
    cur = conn.cursor()
    return conn, cursor


def safe_q(q, return_conn=False, **kwargs):
    conn, cur = get_conn()
    try:
        cur.execute(q, **kwargs)
        if return_conn:
            return conn, cur
        else:
            return cur
    except ps.OperationalError as e:
        sql_error(e)
        return


def safe_q1(q, **kwargs):
    return safe_q(q, **kwargs).fetchone()


def safe_qall(q, **kwargs):
    return safe_q(q, **kwargs).fetchall()


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
        retur


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
    n = min(n, 10000) # no number larger than 10_000
    select = "open, close, high, low, timestamp, volume" if not verbose else "*"
    where = ''

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
    where = add_clause(where, 'period', "period <= %(period)s")
    where = add_clause(where, 'trading_pair', "trading_pair=%(trading_pair)s")

    q = f"""
        SELECT {select} FROM candlesticks
        {where}
        LIMIT {n} 
        ORDER BY timestamp desc;
        """
    results = safe_qall(q, info)
    columns = get_table_columns('candlesticks') if select == "*" else ["open", "close", "high", "low", "timestamp", "volume"]
    df = pd.DataFrame(resluts, columns=columns)
    return df


def get_api(api):
    q = "SELECT * FROM candlesticks WHERE api = %(api)s"
    safe_qall(q, {'api': api})


def remove_api(api):
    """
    Drop API from candle table
    """
    conn, cur = get_conn()
    q = """DELETE FROM candlesticks
           WHERE api = %(api)s"""
    conn, cur = safe_q(q, {'api' : api}, return_conn=True)
    if conn is not None:
        conn.commit()

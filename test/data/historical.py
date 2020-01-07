import os
import time
import cryptolytic.data.historical as h
import cryptolytic.data.sql as sql
import concurrent.futures as futures


def live_update_test():
    # make sure we are in the test database
    assert (os.getenv('POSTGRES_DBNAME')=='cryptotestdb')
    sql.drop_candle_table()
    sql.create_candle_table()
    h.live_update()
    assert len(sql.get_some_candles({}, n=1000), verbose=True) >= 1000
#    with timeout(minutes



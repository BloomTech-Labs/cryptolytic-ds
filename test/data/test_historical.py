import os
import time
import cryptolytic.util.core as util
import cryptolytic.data.historical as h
import cryptolytic.data.sql as sql
import concurrent.futures as futures

from test.start import init
init()  # set enviornemnt variable

def _assert(x):
    assert x

def test_live_update():
    # make sure we are in the test database
    assert (os.getenv('POSTGRES_DBNAME') == 'cryptotestdb')
    sql.drop_candle_table()
    sql.create_candle_table()
    now = time.time()

    def live_update_check():
        assert len(sql.get_some_candles({}, n=1000), verbose=True) >= 1000

    util.timeout(h.live_update, 20, success_handler=live_update_check)  # collect data for 10 seconds
    assert (time.time() - now) >= 20

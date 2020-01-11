import cryptolytic.data.sql as sql
import cryptolytic.util.core as util
import cryptolytic.data.historical as h

def test_check_tables():
    util.timeout(h.live_update, 10)
    for api, exchange_id, trading_pair in h.yield_unique_pair():
        df = (sql.get_some_candles
                ({'api': api, 'exchange_id': exchange_id, 'trading_pair': trading_pair,
                  'period' : 300},
                 n=100000,
                 verbose=True))
        assert df.shape[0] > 100 # check to see that every trading pair has candles for it

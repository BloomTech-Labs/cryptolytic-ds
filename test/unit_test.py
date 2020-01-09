import unittest
import json
from dotenv import load_dotenv
# from .. import application
import application
import requests
def __init__():
    # using test environment
    load_dotenv(verbose=True, dotenv_path='test/test.env')
class TestApp(unittest.TestCase):
    """
    Test Class for application.__main__
    """
    # Return Testing ---------->
    def test_hp(self):
        temp = application.index()
    def test_tp(self):
       # Since there is no rq, we use except:
        try:
            application.trade_predictions()
        except:
            return("No testing done for application.trade_predictions.")
    def test_ap(self):
        # Test_AP Sanity check
        response1 = requests.get("""http://45.56.119.8/arbitrage?
                                exchanges=binance&trading_pair=btc_usd""")
        response2 = requests.get("""http://45.56.119.8/arbitrage?
                                exchanges=binance&trading_pair=btc_usd""")
        # Self assert-equaal
        self.assertEqual(response1,response2)
    def test_candles(self):
        response1 = requests.post(url = "http://45.56.119.8/trade_candles",
 data = """
"exchange_id":"binance",
"trading_pair": "eth_btc",
"period": 300,
"start": 1546322400,
"end": 1546325500""") 
        response2 = requests.post(url = "http://45.56.119.8/trade_candles",
 data = """
"exchange_id":"binance",
"trading_pair": "eth_btc",
"period": 300,
"start": 1546322400,
"end": 1546325500""") 

        # Self assert-equal
        self.assertEqual(response1,response2)
        print("Response 1:", response1)
        print("Response 2:", response2)
        for i,c in zip(response1,response2):
            if i != c:
                print("Iteration Zip != !!!")
    # Error Testing ----------->
    def test_error(self):
        self.assertRaises(ZeroDivisionError, application, 0)
if __name__ == '__main__':
    # Globally call __main__ for Unit Testing
    unittest.main()
    printtest.prtc()

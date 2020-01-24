from flask import Flask, request, render_template, jsonify, send_from_directory
from cryptolytic.data import sql, historical
import pandas as pd
# utils file

application = app = Flask(__name__)


@application.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@application.route('/trade', methods=['POST'])
def trade():
    try:
        content = request.get_json()
        if not {'exchange_id', 'trading_pair'}.issubset(content.keys()):
            return "Error: Require keys exchange_id and trading_pair"

        exchange = content['exchange_id']
        trading_pair = content['trading_pair']
        print(exchange)
        
        # Think have future predictions be updated periodically,
        # and this return the latest predictions
        preds = pd.read_csv('models/future_preds_trade.csv')
        preds = preds[(preds.exchange == exchange) & (preds.trading_pair == trading_pair)]
        return preds.to_json(orient='records')
    except Exception as e:
        return f'Error {e}'


# maybe not necessary with the predictions returned from
# the /trade endpoint
# @application.route('/arbitrage', methods=['POST'])
# def arbitrage():
#     """In: {'from_exchange' : 'binance', 
#             'to_exchange' : 'hitbtc',
#             'trading_pair' : ['btc_usd'],
#             'period': 300}
#        Out: {buy_exchange:, sell_exchange:, }
#        """
#     try:
#         content = request.get_json()
#         return jsonify(content)
#     except Exception as e:
#         return f'Error {e}'


@application.route('/data/<path:path>')
def data_folder(path):
    "Serving static files from the data folder"
    return send_from_directory('data', path)

if __name__ == "__main__":
    application.run(port=8000, debug=True)

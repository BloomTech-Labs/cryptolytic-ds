from flask import Flask, request, render_template, jsonify
from cryptolytic.data import sql, historical
# utils file

application = app = Flask(__name__)

@application.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@application.route('/trade', methods=['POST'])
def trade_predictions():
    try:
        content = request.get_json()
        return jsonify(content)
    except Error as e:
        return f'Error {e}'

@application.route('/arbitrage', methods=['POST'])
def arbitrage_predictions():
    """In: {exchanges : ['binance', etc.], 
            trading_pair : ['btc_usd']
       Out: {buy_exchange:, sell_exchange:, }
       """
    try:
        content = request.get_json()
        return jsonify(content)
    except Error as e:
        return f'Error {e}'

@application.route('/trade_candles', methods=['POST'])
def trade_candles():
    """In: {exchange:}"""
    try:
        content = request.get_json()
        content['exchange']
        content['trading_pair']
        content['start']
        content['end']

        return jsonify(content)
    except Error as e:
        return f'Error {e}'

@application.route('/live_update', methods=['GET'])
def live_update():
    """In: {exchange:}"""
    try:
        print('something')
        historical.live_update()
        return 'thing'
    except Exception as e:
        return f'Error {e}'

if __name__ == "__main__":
    application.run(port=8000, debug=True)

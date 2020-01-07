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
    except Exception as e:
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
    except Exception as e:
        return f'Error {e}'


@application.route('/trade_candles', methods=['POST'])
def trade_candles():
    """In: {exchange:}"""
    try:
        content = request.get_json()
        print(set(content.keys()))
        assert (set(content.keys())
                .issubset({'exchange_id', 'trading_pair', 'period', 'start', 'end'}))
        print("aoetnuhtn")
        df = sql.get_some_candles(content, n=1000000)
        print(df.to_json())
        return df.to_json() # jsonify(content)
    except Exception as e:
        print('Error', e)
        return jsonify({'error' : repr(e)}), 403

if __name__ == "__main__":
    application.run(port=8000, debug=True)

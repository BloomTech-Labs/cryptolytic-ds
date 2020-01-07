from flask import Flask, request, render_template, jsonify, send_from_directory
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
        assert (set(content.keys())
                .issubset({'exchange_id', 'trading_pair', 'period', 'start', 'end'}))
        df = sql.get_some_candles(content, n=1000000)
        return df.to_json()
    except Exception as e:
        print('Error', e)
        return jsonify({'error' : repr(e)}), 403

@application.route('/data/<path:path>')
def data_folder(path):
    "Serving static files from the data folder"
    return send_from_directory('data', path)

if __name__ == "__main__":
    application.run(port=8000, debug=True)

def get_higher_closing(df1, df2):
    # true if df is higher
    categories = (df1['close'] - df2['close'])
    print('something')
#    categories =


def get_higher_closing_test():
    df1 =


# function to create column showing percentage by which higher price is higher
def get_pct_higher(df):
    # i.e., if exchange 1 has a higher closing price than exchange 2...
    if df['higher_closing_price'] == 1:
        # return the percentage by which the exchange 1 closing price is
        # greater than the exchange 2 closing price
        return ((df['close_exchange_1'] /
                 df['close_exchange_2'])-1)*100
    # otherwise, if exchange 2 has a higher closing price than exchange 1...
    elif df['higher_closing_price'] == 2:
        # return the percentage by which the exchange 2 closing price is
        # greater than the exchange 1 closing price
        return ((df['close_exchange_2'] /
                 df['close_exchange_1'])-1)*100
    # otherwise, i.e., if the closing prices are equivalent...
    else:
        # return zero
        return 0


# function to create column showing available arbitrage opportunities
def get_arbitrage_opportunity(df):
    # assuming the total fees are 0.55%, if the higher closing price is less
    # than 0.55% higher than the lower closing price...
    if df['pct_higher'] < .55:
        # return 0, for no arbitrage
        return 0
    # otherwise, if the exchange 1 closing price is more than 0.55% higher
    # than the exchange 2 closing price...
    elif df['higher_closing_price'] == 1:
        # return -1, for arbitrage from exchange 2 to exchange 1
        return -1
    # otherwise, if the exchange 2 closing price is more than 0.55% higher
    # than the exchange 1 closing price...
    elif df['higher_closing_price'] == 2:
        # return 1, for arbitrage from exchange 1 to exchange 2
        return 1

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cryptolytic.data.historical as h
from cryptolytic.start import init

def arb():
    # get (normalized/feature engineered) data and model get model for a given trading pair
    # train test split
    # two week gap between train and test sets to check model performance (maybe not)
    # load model, prun predictiotn s
    pass

def trade_model(df):
    # load data 
    max_depth = 17
    max_features = 40

    # get data with feature engineering
    # check that the data is valid (no empty or small dataframes)
    # train test split 

    train_size = int(train.shape[0] * 0.8)
    train = df.iloc[0:train_size]
    test = df.iloc[train_size:]


    features = df.columns[0:50] # first 50 columns as features for right now
    target = 'price_increased'
    # define X, y vectors
    X_train = train[features]
    X_test = test[features]
    y_train = train[target]
    y_test = test[target]

    # Random forest classifier
    model = RandomForestClassifier(max_features=max_features, 
                           max_depth=max_depth,
                           n_estimators=100,
                           n_jobs=-1,
                           random_state=42)
    
    model.fit(X_train, y_train)
    train_score=  model.score(X_train, y_train)
    y_preds = model.predict(X_test)
    score = accuracy_score(y_test, y_preds)

    return y_preds, score

    # Get profit and loss
    # 

def test_trade_model():
    init()
    exchange_id = 'hitbtc'
    trading_pair = 'eos_usd'
    period = 300
    start = '2019-01-30'
    
    df = h.get_data(exchange_id, trading_pair, period, start, n=8000)
    preds, score = trade_model(df)

    print(score)
    print(preds)


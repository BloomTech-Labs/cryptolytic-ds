from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_scorer

def arb():
    # get (normalized/feature engineered) data and model get model for a given trading pair
    # train test split
    # two week gap between train and test sets to check model performance (maybe not)
    # load model, prun predictiotn s
    pass

def trade_model(df):
    # load data 
    max_depth = 17
    max_feature_list = 40

    # get data with feature engineering
    # check that the data is valid (no empty or small dataframes)
    # train test split 

    train = df[df.index < '2019-01-30 23:00:00']
    test = df[(df.index > '2019-03-01 23:00:00') & (df.index < '2019-05-01 23:00:00')] 

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

    # Get profit and loss
    # 

def test_trade_model():

    pass

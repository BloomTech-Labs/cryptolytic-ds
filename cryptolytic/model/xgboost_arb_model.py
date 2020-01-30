from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cryptolytic.data.historical as h
from cryptolytic.start import init


def data_splice(df):
    '''
    Funciton splices data into x and y train and test
    Requires dataframe of data
    Returns x and y train and test
    '''

    train_size = int(df.shape[0] * 0.8)
    train = df.iloc[0:train_size]
    test = df.iloc[train_size:]

    features = df.columns[0:50]  # first 50 columns as features for right now
    # Categorical target for the model to achieve
    # 1 = price increase
    # 0 = no price increase
    # 2 = price decrease # TODO Implemant this expanded category
    target = 'arb_signal_class'
    # define X, y vectors
    X_train = train[features]
    X_test = test[features]
    y_train = train[target]
    y_test = test[target]

    return X_train, y_train, X_test, y_test


def create_model(params={}):
    # load data
    max_depth = 17
    max_features = 40

    # Random forest classifier
    model = RandomForestClassifier(max_features=max_features,
                                   max_depth=max_depth,
                                   n_estimators=100,
                                   n_jobs=-1,
                                   random_state=42)
    return model


def fit_model(model, x_train, y_train):
    # Fit the given model with the x_train and y_train data
    model.fit(x_train, y_train)
    # Return the fitted model
    return model


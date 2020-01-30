from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cryptolytic.data.historical as h
from cryptolytic.start import init


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


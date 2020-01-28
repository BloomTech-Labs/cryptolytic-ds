# begin imports
from __future__ import print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
# from scipy.stats import yeojohnson
import os
from cryptolytic.util import *
from kerastuner.tuners import RandomSearch

# tensorflow import
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Activation,\
    Add, Input, LSTM
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.constraints as constraints
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.initializers import glorot_uniform, glorot_uniform

# sklearn imports
from sklearn.preprocessing import MinMaxScaler


def stacked_dataset(models, inputX):
    stackX = None
    for model in models:
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilties]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX


def stacked_prediction(models, model, inputX):
    stackedX = stacked_dataset(models, inputX)
    yhat = model.predict(stackedX)
    return yhat


def endingly():
    models, params = load_all_models('example')
    model = fit_stacked_model(models, x_train, y_train)
    preds = stacked_prediction(models, model, x_train)
    evaluate_models(models)
    return preds


def hyperparameter(inputX, inputy):
    filtershape1 = 32 + 16 * np.random.randint(0, 6)
    filtershape2 = 32 + 16 * np.random.randint(0, 6)

    params = adict(
        filters1=32 + 16 * np.random.randint(0, 5),
        noise1=np.random.uniform(high=0.01),
        filtershape1=[filtershape1, filtershape1, filtershape1*2],
        filtershape2=[filtershape2, filtershape2, filtershape2*2]
    )
    print(params)
    model = create_model(inputX, params)
    fit_model(model, inputX, inputy)
    save_model(model, 'filter', params=params)
    return model


def run_tuning():
    models = []
    for i in range(50):
        models.append(hyperparameter(x_train, y_train))
    return models


def evaluate_models(models):
    for m in models:
        _, acc = model.evaluate(x_val, y_val)


def random_keras_tuner(compiled_model, objective='val_accuracy', max_trials=5,
                       executions_per_trial=3):
    tuner = RandomSearch(
        compiled_model,
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='cryptolytic-ds'
        project_name='cryptolytic'
    )
    tuner.results_summary()
    return tuner

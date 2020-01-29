'''
def get_model_path(api, exchange_id, trading_pair):
    return f'models/models/model_{api}_{exchange_id}_{trading_pair}.h5'
}
'''
# begin imports

# internal imports
from cryptolytic.start import init
import cryptolytic.data as d
import cryptolytic.model.data_work as dw
import cryptolytic.model.model_framework as mfw
import cryptolytic.model.hyperparameter as hyper

# external imports
import ta
import os


def cron_train(api, exchange_id, traiding_pair):
    '''
    Funciton is created to allow for the creation of new/updated models by
    providing api call information.
    Requires
    '''
    # initialize the enviornment
    init()

    # get current time
    now = time.time()
    now = int(now)
    # create default values
    history_size = 400
    input_len = 8000
    lahead = 12*3
    step = 2
    period = 300
    to_drop = lahead - 1
    input_len = input_len + to_drop
    batch_size = 200
    # calculate the start time to pull data from
    start_time = now - input_len*period

    # grab data from the database
    df_original = d.get_df(
        {'start': start_time, 'period': period, 'trading_pair': traiding_pair,
         'exchange_id': exchange_id},
        n=input_len
    )

    # Create the Primary dataframe
    df = df_original
    df = df.sort_index()
    # Make sure the data frame is numeric data only.
    # 'period' column is dropped as all values of 'period' should be the same
    df = df.__get_numeric_data().drop(['period'], axis=1, errors='ignore')
    # Filter out the 'timestamp_* metrics
    df = df.filter(regex='(?!timestamp_.*)', axis=1)
    # Create signals from data using the Technical Analysis library
    df = ta.add_all_ta_features(
        df, open='open', high='high', low='low', close='close', volume='volume'
        ).dropna(axis=1)
    #
    df_diff = (df - df.shift(1, fill_value=0)).\
        rename(lambda x: x+'_diff', axis=1)
    # concatinate df_diff into the primary dataframe (df)
    df = pd.concat([df, df_diff], axis=1)
    # Normalize the dataset to allow for accurate training and prediction
    dataset = dw.normalize(df.values)
    # Create the target for the model to train towards
    target_column = df.columns.get_loc('close')

    # Create the windowed x and y train and test sets
    x_train, y_train, x_val, y_val = dw.windowed(
        dataset, target_column, batch_size, history_size, step, lahead
    )

    # Create a hyperparameter tunned model
    '''
    hyper.hyperparameter requires x and y train and val data and automatically
    performs hyperparameter tuning, model creation, and model saving
    '''
    hyper_model, params = hyper.hyperparameter(x_train, y_train, x_val, y_val)

    # Save the model to the folder
    path = 'models/models/'
    filename = os.path.join(
        path, f'model_{api}_{exchange_id}_{trading_pair}.h5'
        )
    model.save(filename)
    # Save the parameters to the folder
    param_path = os.path.join(
        path, f'model_{api}_{exchange_id}_{trading_pair}', '_params.csv'
        )
    pd.DataFrame(params).to_csv(param_path)

    print('Model saved')

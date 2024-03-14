TIMESTEPS=12
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


train_proportion=0.7
valid_proportion=0.9

store=2
brand = 'tropicana'

def create_dataset(data, timesteps=TIMESTEPS):
    """Create input and output pairs for training lstm.
    Params:
        data (pandas.DataFrame): Normalized dataset
        timesteps (int, default: TIMESTEPS): Input time length
    Returns:
        X (numpy.array): Input for lstm
        y (numpy.array): Output for lstm
        y_date (list): Datetime of output
        start_values (list): Start valeus of each input
    """
    X, y, y_date, start_values = [], [], [], []

    for i in range(len(data) - timesteps):
        Xt = data.iloc[i:i + timesteps].values
        yt = data.iloc[i + timesteps]
        yt_date = data.iloc[i + timesteps]

        # Subtract a start value from each values in the timestep.
        start_value = Xt[0]
        Xt = Xt - start_value
        yt = yt - start_value

        X.append(Xt)
        y.append(yt)
        y_date.append(yt_date)
        start_values.append(start_value)

    return np.array(X), np.array(y), y_date, start_values


def split_train_valid_test(X, y, time,train_proportion=train_proportion,valid_proportion=valid_proportion,timesteps=TIMESTEPS):#, train_range=TRAIN_RANGE, valid_range=VALID_RANGE, test_range=TEST_RANGE
    """Split X and y into train, valid, and test periods.
    Params:
        X (numpy.array): Input for lstm
        y (numpy.array): Output for lstm
        y_date (list): Datetime of output
        train_range (tuple): Train period
        valid_range (tuple): Validation period
        test_range (tuple): Test period
    Returns:
        X_train (pandas.DataFrame)
        X_valid (pandas.DataFrame)
        X_test (pandas.DataFrame)
        y_train (pandas.DataFrame)
        y_valid (pandas.DataFrame)
        y_test (pandas.DataFrame)
        y_date_train (list)
        y_date_valid (list)
        y_date_test (list)
    """
    # train_end_idx = y_date.index(train_range[1])
    # valid_end_idx = y_date.index(valid_range[1])

    train_end_idx =int(X.shape[0] * train_proportion)
    valid_end_idx =int(X.shape[0] * valid_proportion)
    X_train = X[:train_end_idx + 1, :]
    X_valid = X[train_end_idx + 1:valid_end_idx + 1, :]
    X_test = X[valid_end_idx + 1:, :]

    y_train = y[:train_end_idx + 1]
    y_valid = y[train_end_idx + 1:valid_end_idx + 1]
    y_test = y[valid_end_idx + 1:]

    time=time.iloc[timesteps:]
    time_train = time.iloc[:train_end_idx + 1]
    time_valid = time.iloc[train_end_idx + 1:valid_end_idx + 1]
    time_test = time.iloc[valid_end_idx + 1:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test, time_train, time_valid, time_test

def get_date(time,timesteps=TIMESTEPS,train_proportion=train_proportion,valid_proportion=valid_proportion):
    time=time.iloc[timesteps:]
    train_end_idx =int(time.shape[0] * train_proportion)
    valid_end_idx =int(time.shape[0] * valid_proportion)
    time_train = time.iloc[:train_end_idx + 1]
    time_valid = time.iloc[train_end_idx + 1:valid_end_idx + 1]
    time_test = time.iloc[valid_end_idx + 1:]
    return time_train,time_valid,time_test

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

def evaluate_model(data, scaler, X_train, X_valid, X_test, y_train, y_valid, y_test, start_values, model):
    """Evaluate trained model by rmse (root mean squared error) and mae (mean absolute error)'"""

    # Predict next month passengers
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    # Add start_values that were subtracted when preprocessing.
    pred_train = pred_train + start_values[:len(X_train)]
    pred_valid = pred_valid + start_values[len(X_train):len(X_train) + len(X_valid)]
    pred_test = pred_test + start_values[len(X_train) + len(X_valid):]

    # Inverse transform normalization
    pred_train = scaler.inverse_transform(pred_train).flatten()
    pred_valid = scaler.inverse_transform(pred_valid).flatten()
    pred_test = scaler.inverse_transform(pred_test).flatten()

    # Add start_values that were subtracted when preprocessing.
    y_train = y_train + start_values[:len(X_train)]
    y_valid = y_valid + start_values[len(X_train):len(X_train) + len(X_valid)]
    y_test = y_test + start_values[len(X_train) + len(X_valid):]

    # Inverse transform normalization
    y_train = scaler.inverse_transform(y_train).flatten()
    y_valid = scaler.inverse_transform(y_valid).flatten()
    y_test = scaler.inverse_transform(y_test).flatten()

    # Evaluate prediction scores of model.
    # for y, pred, mode in zip([y_train, y_valid, y_test], [pred_train, pred_valid, pred_test],
    #                          ['train', 'valid', 'test']):
    #     # rmse = np.sqrt(mean_squared_error(y, pred))
    #     # mae = mean_absolute_error(y, pred)

    MAPE_train = mape(y_train, pred_train)
    MAPE_valid = mape(y_valid, pred_valid)
    MAPE_test = mape(y_test, pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_valid = np.sqrt(mean_squared_error(y_valid, pred_valid))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_valid = mean_absolute_error(y_valid, pred_valid)
    mae_test = mean_absolute_error(y_test, pred_test)
    return (MAPE_train,MAPE_valid,MAPE_test,rmse_train,rmse_valid,rmse_test,mae_train,mae_valid,mae_test,pred_train,pred_valid,pred_test)
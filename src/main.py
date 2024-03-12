import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from math import ceil, log
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, \
    Dense, Dropout, Embedding, Flatten, Input, LSTM, Lambda, RepeatVector, \
    concatenate, dot
from tensorflow.keras.models import Model


TRAIN_RANGE = (datetime(2021, 1, 4), datetime(2022, 12, 31))
VALID_RANGE = (datetime(2023, 1, 1), datetime(2023, 6, 30))
TEST_RANGE = (datetime(2023, 7,1), datetime(2024, 1, 28))
TIMESTEPS = 12  # Input 12 months to predict next month

# data = pd.read_csv('data/female-births.csv')
# data.index = pd.to_datetime(data.date)  # Set datetime index
# data.drop(['date'], axis=1, inplace=True)
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# data['NormalizedPassengers'] = scaler.fit_transform(data['quantity'].values.reshape(-1, 1)).flatten()

selected_data = pd.read_csv('data/customer_data.csv')
# 转换日期列为日期时间格式
selected_data['delivery_date'] = pd.to_datetime(selected_data['delivery_date'])
# 设置索引为日期列
# 选择特定的SKU ID
sku_id = 'BK-1320'
data = selected_data[selected_data['sku_id'] == sku_id]
# 重置索引，以便日期成为列
data.reset_index(inplace=True)
# 输出结果
#print(data)

data.index = pd.to_datetime(data.delivery_date)
data.drop(['delivery_date'], axis=1, inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data['NormalizedPassengers'] = scaler.fit_transform(data['total_qty'].values.reshape(-1, 1)).flatten()


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
        yt_date = data.index[i + timesteps].to_pydatetime()

        # Subtract a start value from each values in the timestep.
        start_value = Xt[0]
        Xt = Xt - start_value
        yt = yt - start_value

        X.append(Xt)
        y.append(yt)
        y_date.append(yt_date)
        start_values.append(start_value)

    return np.array(X), np.array(y), y_date, start_values


def split_train_valid_test(X, y, y_date, train_range=TRAIN_RANGE, valid_range=VALID_RANGE, test_range=TEST_RANGE):
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
    train_end_idx = y_date.index(train_range[1])
    valid_end_idx = y_date.index(valid_range[1])

    X_train = X[:train_end_idx + 1, :]
    X_valid = X[train_end_idx + 1:valid_end_idx + 1, :]
    X_test = X[valid_end_idx + 1:, :]

    y_train = y[:train_end_idx + 1]
    y_valid = y[train_end_idx + 1:valid_end_idx + 1]
    y_test = y[valid_end_idx + 1:]

    y_date_train = y_date[:train_end_idx + 1]
    y_date_valid = y_date[train_end_idx + 1:valid_end_idx + 1]
    y_date_test = y_date[valid_end_idx + 1:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid, y_date_test


# Create input and output pairs for training lstm.
X, y, y_date, start_values = create_dataset(data[['NormalizedPassengers']])

# Split X and y into train, valid, and test periods.
X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid, y_date_test = split_train_valid_test(X,
                                                                                                                     y,
                                                                                                                     y_date)
print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)

def create_model(timesteps=TIMESTEPS):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(timesteps, 1), name='lstm_1'))  # Input timesteps months with scalar value.
    model.add(LSTM(32, name='lstm_2'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.01), metrics=['mean_absolute_error'])
    return model
# Create model

def create_LSTM_model(
        max_cat_id,
        seq_len=12,

        n_dyn_fea=1,
        n_outputs=2,
        latent_dim=128,

):
    # latent_dim = 8, 41
    encoder_inputs = Input(shape=(seq_len, n_dyn_fea))

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, seq_len, n_dyn_fea),
                   stateful=False,
                   return_sequences=False,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # state_h = BatchNormalization(momentum=0.6)(state_h)
    # state_c = BatchNormalization(momentum=0.6)(state_c)
    decoder_inputs = RepeatVector(n_outputs)(state_h)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, n_dyn_fea + 0),
                     stateful=False,
                     return_sequences=True,
                     return_state=True,
                     dropout=0.2,
                     recurrent_dropout=0.2)

    decoder_outputs, _, _ = decoder_1(
        decoder_inputs, initial_state=[state_h, state_c])
    decoder_outputs = Flatten()(decoder_outputs)

    n_cat_fea = len(max_cat_id)
    cat_fea_in = Input(shape=(n_cat_fea,), dtype="uint8")
    cat_flatten = []
    for i, m in enumerate(max_cat_id):
        cat_fea = Lambda(lambda x, i: x[:, i, None], arguments={
            "i": i})(cat_fea_in)
        cat_fea_embed = Embedding(
            m + 1, ceil(log(m + 1)), input_length=1)(cat_fea)
        cat_flatten.append(Flatten()(cat_fea_embed))

    d = concatenate([decoder_outputs] + cat_flatten)
    d = Dense(16, activation="relu")(d)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=[encoder_inputs, cat_fea_in], outputs=output)
    return model

model = create_LSTM_model(max_cat_id=[1])
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.01), metrics=['mean_absolute_error'])
model.summary()

es = EarlyStopping(monitor='val_mean_absolute_error', min_delta=0, patience=15, verbose=1, mode='auto')
fn = 'model/cusdata_trained_model.h5'
mc = ModelCheckpoint(filepath=fn, save_best_only=True)
callbacks = [es, mc]


# # 模型训练
# model.fit(x=[encoder_inputs_data, cat_fea_in], y=output_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)#encoder输入是12,1 cat_fea是整个数据长度

# Start training model.
fit = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=200,
    verbose=2,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks)

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
axL.plot(fit.history['loss'], label="loss (mse) for training")
axL.plot(fit.history['val_loss'], label="loss (mse) for validation")
axL.set_title('model loss')
axL.set_xlabel('epoch')
axL.set_ylabel('loss')
axL.legend(loc='upper right')

axR.plot(fit.history['mean_absolute_error'], label="mae for training")
axR.plot(fit.history['val_mean_absolute_error'], label="mae for validation")
axR.set_title('model mse')
axR.set_xlabel('epoch')
axR.set_ylabel('mse')
axR.legend(loc='upper right')

plt.show()

# Load best model
#model = load_model(fn)
# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def mape(labels, predicts, mask):
    """
        Mean absolute percentage. Assumes ``y >= 0``.
        Defined as ``(y - y_pred).abs() / y.abs()``
    """
    loss = np.abs(predicts - labels) / (np.abs(labels) + 1)
    loss *= mask
    non_zero_len = mask.sum()
    return np.sum(loss)/non_zero_len


def evaluate_model(data, scaler, X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid,
                   y_date_test, start_values, model):
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

    pred_df = data.copy()
    pred_df.loc[y_date_train[0]:y_date_train[-1], 'PredictionTrain'] = pred_train
    pred_df.loc[y_date_valid[0]:y_date_valid[-1], 'PredictionValid'] = pred_valid
    pred_df.loc[y_date_test[0]:y_date_test[-1], 'PredictionTest'] = pred_test
    pred_df[['total_qty', 'PredictionTrain', 'PredictionValid', 'PredictionTest']].plot(figsize=(12, 6),
                                                                                         title='Predicted monthly airline passengers')
    plt.show()

    # Add start_values that were subtracted when preprocessing.
    y_train = y_train + start_values[:len(X_train)]
    y_valid = y_valid + start_values[len(X_train):len(X_train) + len(X_valid)]
    y_test = y_test + start_values[len(X_train) + len(X_valid):]

    # Inverse transform normalization
    y_train = scaler.inverse_transform(y_train).flatten()
    y_valid = scaler.inverse_transform(y_valid).flatten()
    y_test = scaler.inverse_transform(y_test).flatten()

    # Evaluate prediction scores of model.
    for y, pred, mode in zip([y_train, y_valid, y_test], [pred_train, pred_valid, pred_test],
                             ['train', 'valid', 'test']):
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)

        real_y_true_mask = (1 - (y == 0))
        MAPE = mape(y, pred, real_y_true_mask)
        #MAPE = mape(y, pred)
        print(f'{mode} rmse: {rmse:.06f}, mae: {mae:.06f}，mape: {MAPE:.06f}')


evaluate_model(data, scaler, X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid,
               y_date_test, start_values, model)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
#from datetime import datetime
import matplotlib.pyplot as plt
from math import ceil, log
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
#tf.random.set_seed(0)
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, \
    Dense, Dropout, Embedding, Flatten, Input, LSTM, Lambda, RepeatVector, \
    concatenate, dot
from tensorflow.keras.models import Model
# from openpyxl import load_workbook
# from openpyxl import Workbook
import pymongo
import uuid
import dataprocessing
# TRAIN_RANGE = (datetime(2021, 1, 4), datetime(2022, 12, 31))
# VALID_RANGE = (datetime(2023, 1, 1), datetime(2023, 6, 30))
# TEST_RANGE = (datetime(2023, 7,1), datetime(2024, 1, 28))
TIMESTEPS = 12  # Input 12 months to predict next month

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
        #yt_date = data.week
        #yt_date = data.index[i + timesteps].to_pydatetime()

        # Subtract a start value from each values in the timestep.
        start_value = Xt[0]
        Xt = Xt - start_value
        yt = yt - start_value

        X.append(Xt)
        y.append(yt)
        #y_date.append(yt_date)
        start_values.append(start_value)

    return np.array(X), np.array(y), y_date, start_values


def split_train_valid_test(X, y, time,timesteps=TIMESTEPS):#, train_range=TRAIN_RANGE, valid_range=VALID_RANGE, test_range=TEST_RANGE

    # train_end_idx = y_date.index(train_range[1])
    # valid_end_idx = y_date.index(valid_range[1])

    train_end_idx =int(X.shape[0] * 0.7)
    valid_end_idx =int(X.shape[0] * 0.9)
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

# Create input and output pairs for training lstm.

def create_LSTM_model0(
        seq_len=12,  # Lookback window of the lagged dynamic features
        n_outputs=1,  # How many timesteps of outputs to produce
        n_dilated_layers=3,
        kernel_size=2,
        n_filters=3,
        dropout_rate=0.1):
    """Create a Dilated CNN model.
    Args:
        seq_len (int): Input sequence length
        n_outputs (int): Number of outputs of the network
        kernel_size (int): Kernel size of each convolutional layer
        n_filters (int): Number of filters in each convolutional layer
        dropout_rate (float): Dropout rate in the network
    Returns:
        object: Keras Model object
    """
    # Sequential input for dynamic features
    seq_in = Input(shape=(seq_len, 1))

    # Dilated convolutional layers
    conv1d_layers = []
    conv1d_layers.append(
        Conv1D(filters=n_filters, kernel_size=kernel_size,
               dilation_rate=1, padding="causal", activation="relu")(
            seq_in)
    )
    for i in range(1, n_dilated_layers):
        conv1d_layers.append(
            Conv1D(
                filters=n_filters, kernel_size=kernel_size,
                dilation_rate=2 ** i, padding="causal",
                activation="relu"
            )(conv1d_layers[i - 1])
        )

    # Skip connections
    if n_dilated_layers > 1:
        c = concatenate([conv1d_layers[0], conv1d_layers[-1]])
    else:
        c = conv1d_layers[0]

    # Output of convolutional layers
    conv_out = Conv1D(8, 1, activation="relu")(c)
    conv_out = Dropout(dropout_rate)(conv_out)
    conv_out = Flatten()(conv_out)

    x = conv_out

    x = Dense(16, activation="relu")(x)
    output = Dense(n_outputs, activation="linear")(x)

    # Define model interface
    inputs = seq_in

    model = Model(inputs=inputs, outputs=output)

    return model

def create_LSTM_model_seq1(shape1, shape2):
    latent_dim=128
    n_outputs=1
    model = Sequential()
    # Encoder LSTM layer
    model.add(LSTM(latent_dim,
                   batch_input_shape=(1, shape1, shape2),
                   stateful=False,
                   return_sequences=False,
                   return_state=True,
                   recurrent_initializer='glorot_uniform'))

    # RepeatVector layer for decoder input
    model.add(RepeatVector(n_outputs))
    # Decoder LSTM layer
    model.add(LSTM(latent_dim,
                   batch_input_shape=(1, n_outputs, shape2 + 0),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   dropout=0.2,
                   recurrent_dropout=0.2))
    # Flatten layer
    model.add(Flatten())
    # Dense layers
    model.add(Dense(16, activation="relu"))
    model.add(Dense(n_outputs, activation="linear"))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])
    return model

def create_LSTM_model1(
        seq_len=12,
        n_dyn_fea=1,
        n_outputs=1,
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
    d = decoder_outputs
    d = Dense(16, activation="relu")(d)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)
    return model

from keras.layers import Input, LSTM, RepeatVector, Dense, Flatten
from keras.models import Model

def create_LSTM_model2(seq_len=12,
                       n_outputs=1,
                       latent_dim=128
                       ):
    encoder_inputs = Input(shape=(seq_len, 1))

    encoder = LSTM(latent_dim,
                   batch_input_shape=(1, seq_len, 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = RepeatVector(n_outputs)(state_h)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, 1),
                     stateful=False,
                     return_sequences=True,
                     return_state=False,
                     dropout=0.2,
                     recurrent_dropout=0.2)
    decoder_2 = LSTM(120,
                     stateful=False,
                     return_sequences=True,
                     return_state=True,
                     dropout=0.2,
                     recurrent_dropout=0.2)

    decoder_outputs, _, _ = decoder_2(
        decoder_1(decoder_inputs, initial_state=encoder_states))
    decoder_outputs = Flatten()(decoder_outputs)

    d = decoder_outputs
    d = Dense(16, activation="relu")(d)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)

    return model

def create_LSTM_model3( seq_len=12,
                       n_dyn_fea=1,
                       n_outputs=1,
                       latent_dim=60):
    # 1层LSTM decoder, with attention
    # latent_dim=8 38
    encoder_inputs = Input(shape=(seq_len, n_dyn_fea))

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, seq_len, n_dyn_fea + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    _, state_h, state_c = encoder(encoder_inputs)
    # state_h = BatchNormalization(momentum=0.6)(state_h)
    # state_c = BatchNormalization(momentum=0.6)(state_c)
    encoder_states = [state_h, state_c]

    decoder_inputs = RepeatVector(n_outputs)(state_h)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, n_dyn_fea + 0),
                     stateful=False,
                     return_sequences=True,
                     return_state=False,
                     dropout=0.2,
                     recurrent_dropout=0.2)
    decoder_outputs = decoder_1(decoder_inputs, initial_state=encoder_states)

    encoder_outputs = RepeatVector(n_outputs)(state_h)

    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)

    context = dot([attention, encoder_outputs], axes=[2, 1])
    context = BatchNormalization(momentum=0.6)(context)

    decoder_outputs = concatenate([context, decoder_outputs])
    decoder_outputs = Flatten()(decoder_outputs)

    d = decoder_outputs
    d = Dense(16, activation="relu")(d)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)
    return model

def create_LSTM_model4(
                       seq_len=12,
                       n_dyn_fea=1,
                       n_outputs=1,
                       latent_dim=60):
    encoder_inputs = Input(shape=(seq_len, n_dyn_fea))
    k = 4

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, seq_len, n_dyn_fea + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    w = Dense(k, use_bias=False)(state_h)  # 1 * latent_dim -> 1 * k
    encoder_outputs = Conv1D(filters=k, kernel_size=1)(
        encoder_outputs)  # seq_len * latent_dim -> seq_len * k

    a = dot([encoder_outputs, w], axes=[2, 1])  # seq_len * 1
    a = tf.sigmoid(a)  # seq_len * 1

    v = dot([encoder_outputs, a], axes=[1, 1])  # k * 1
    out = concatenate([state_h, v])
    encoder_outputs = Dense(latent_dim)(out)

    decoder_inputs = RepeatVector(n_outputs)(state_h)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, latent_dim),
                     stateful=False,
                     return_sequences=True,
                     return_state=False,
                     dropout=0.2,
                     recurrent_dropout=0.2)
    decoder_outputs = decoder_1(decoder_inputs, initial_state=[
        encoder_outputs, state_c])

    encoder_outputs = RepeatVector(n_outputs)(state_h)

    # m * latent_dim, m * latent_dim -> m * m
    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)

    # m * m, m * latent_dim -> m * latent_dim
    context = dot([attention, encoder_outputs], axes=[2, 1])
    context = BatchNormalization(momentum=0.6)(context)

    decoder_outputs = concatenate([context, decoder_outputs])
    decoder_outputs = Flatten()(decoder_outputs)

    d = Dense(16, activation="relu")(decoder_outputs)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)

    return model

def create_LSTM_model5(
                       seq_len=12,
                       n_dyn_fea=1,
                       n_outputs=1,
                       latent_dim=60):
    encoder_inputs = Input(shape=(seq_len, n_dyn_fea))
    k = 4

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, seq_len, n_dyn_fea + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_outputs = tf.reshape(
        encoder_outputs, [-1, latent_dim, seq_len])  # latent_dim * seq_len

    w = Dense(k, use_bias=False)(state_h)  # 1 * latent_dim -> 1 * k
    encoder_outputs = Conv1D(filters=k, kernel_size=1)(
        encoder_outputs)  # latent_dim * seq_len -> latent_dim * k

    a = dot([encoder_outputs, w], axes=[2, 1])  # latent_dim * 1
    a = tf.sigmoid(a)  # latent_dim * 1

    v = dot([encoder_outputs, a], axes=[1, 1])  # k * 1

    out = concatenate([state_h, v])

    encoder_outputs = Dense(latent_dim)(out)

    decoder_inputs = RepeatVector(n_outputs)(state_h)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, latent_dim),
                     stateful=False,
                     return_sequences=True,
                     return_state=False,
                     dropout=0.2,
                     recurrent_dropout=0.2)
    decoder_outputs = decoder_1(decoder_inputs, initial_state=[
        encoder_outputs, state_c])

    encoder_outputs = RepeatVector(n_outputs)(state_h)

    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)

    context = dot([attention, encoder_outputs], axes=[2, 1])
    context = BatchNormalization(momentum=0.6)(context)

    decoder_outputs = concatenate([context, decoder_outputs])
    decoder_outputs = Flatten()(decoder_outputs)

    d = Dense(16, activation="relu")(decoder_outputs)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)
    return model

def create_LSTM_model6(
                       seq_len=12,
                       n_dyn_fea=1,
                       n_outputs=1,
                       latent_dim=60):
    encoder_inputs = Input(shape=(seq_len, n_dyn_fea))
    k = 4

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, seq_len, n_dyn_fea + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    e = encoder_outputs[:, :-1, :]
    e = tf.reshape(e, [-1, latent_dim, seq_len - 1])  # latent_dim * seq_len
    # latent_dim * seq_len -> latent_dim * k
    e = Conv1D(filters=k, kernel_size=1)(e)

    f = encoder_outputs  # seq_len * latent_dim
    # seq_len * latent_dim -> seq_len * k
    f = Conv1D(filters=k, kernel_size=1)(f)
    m = Dense(k, use_bias=False)(state_h)  # 1 * latent_dim -> 1 * k
    # seq_len * latent_dim -> seq_len * k
    f = Conv1D(filters=k, kernel_size=1)(f)
    b = dot([f, m], axes=[2, 1])  # seq_len * 1
    b = Dense(latent_dim, use_bias=True)(b)  # latent_dim * 1
    b = tf.sigmoid(b)  # latent_dim * 1

    n = dot([b, e], axes=[1, 1])  # k * 1

    out = concatenate([n, state_h])
    encoder_outputs = Dense(latent_dim)(out)

    # state_h = BatchNormalization(momentum=0.6)(state_h)

    decoder_inputs = RepeatVector(n_outputs)(state_h)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, latent_dim),
                     stateful=False,
                     return_sequences=True,
                     return_state=False,
                     dropout=0.2,
                     recurrent_dropout=0.2)
    decoder_outputs = decoder_1(decoder_inputs, initial_state=[
        encoder_outputs, state_c])

    encoder_outputs = RepeatVector(n_outputs)(state_h)

    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)

    context = dot([attention, encoder_outputs], axes=[2, 1])
    context = BatchNormalization(momentum=0.6)(context)

    decoder_outputs = concatenate([context, decoder_outputs])
    decoder_outputs = Flatten()(decoder_outputs)

    d = Dense(16, activation="relu")(decoder_outputs)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)
    return model

def create_LSTM_model7(
                       seq_len=12,
                       n_dyn_fea=1,
                       n_outputs=1,
                       latent_dim=60):
    encoder_inputs = Input(shape=(seq_len, n_dyn_fea))
    k = 4

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, seq_len, n_dyn_fea + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    w = Dense(k, use_bias=False)(state_h)  # 1 * latent_dim -> 1 * k
    encoder_outputs = Conv1D(filters=k, kernel_size=1)(
        encoder_outputs)  # seq_len * latent_dim -> seq_len * k

    a = dot([encoder_outputs, w], axes=[2, 1])  # seq_len * 1
    a = tf.sigmoid(a)  # seq_len * 1

    v = dot([encoder_outputs, a], axes=[1, 1])  # k * 1
    out = concatenate([state_h, v])
    encoder_outputs = Dense(latent_dim)(out)

    decoder_inputs = Conv1D(filters=latent_dim * 2, kernel_size=2,
                            dilation_rate=1, padding='causal',
                            activation='relu')(encoder_inputs)
    decoder_inputs = Conv1D(filters=latent_dim * 2, kernel_size=2,
                            dilation_rate=2, padding='causal',
                            activation='relu')(
        decoder_inputs)  # seq_len * (latent_dim * 2)
    decoder_inputs = tf.reshape(decoder_inputs, [-1, latent_dim * 2, seq_len])
    decoder_inputs = Conv1D(filters=n_outputs, kernel_size=2, dilation_rate=1,
                            padding='causal',
                            activation='relu')(decoder_inputs)
    decoder_inputs = tf.reshape(
        decoder_inputs, [-1, n_outputs, latent_dim * 2])
    decoder_inputs = Dropout(0.1)(decoder_inputs)
    decoder_inputs = Dense(latent_dim)(decoder_inputs)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, latent_dim),
                     stateful=False,
                     return_sequences=True,
                     return_state=False,
                     dropout=0.2,
                     recurrent_dropout=0.2)
    decoder_outputs = decoder_1(decoder_inputs, initial_state=[
        encoder_outputs, state_c])

    encoder_outputs = RepeatVector(n_outputs)(state_h)

    # m * latent_dim, m * latent_dim -> m * m
    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)

    # m * m, m * latent_dim -> m * latent_dim
    context = dot([attention, encoder_outputs], axes=[2, 1])
    context = BatchNormalization(momentum=0.6)(context)

    decoder_outputs = concatenate([context, decoder_outputs])
    decoder_outputs = Flatten()(decoder_outputs)


    d = decoder_outputs
    d = Dense(16, activation="relu")(d)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs ,outputs=output)

    return model

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

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
    return (MAPE_train,MAPE_valid,MAPE_test,pred_train,pred_valid,pred_test)


store=2
brand = 'tropicana'
myclient = pymongo.MongoClient("mongodb://localhost:27017")
# myclient = pymongo.MongoClient('localhost',27017)
mydb = myclient["mydb1"]  # 指定数据库
mycol = mydb["oj"]  # 指定集合
myuuid= mydb["MAPE"]
mywrite=mydb["test"]

uuid="408f7b76-3a84-47bc-af7a-697e527caabf"

mywrite = mydb[uuid]
data_from_mongodb = list(mywrite.find())
store=dataprocessing.store
brand=dataprocessing.brand
data = pd.DataFrame(data_from_mongodb)
data = data[(data['store'] == store) & (data['brand'] == brand)]

time_train, time_valid, time_test = dataprocessing.get_date(data.week)
print(time_train)


    # 获取对应模型号的预测值的键
pred_train_key = f"pred_train_{1}"  # 假设预测值的键为 pred_test_i，其中 i 为模型号
pred_valid_key = f"pred_valid_{1}"
pred_test_key = f"pred_test_{1}"







# # generate UUID
# uuid_value = str(uuid.uuid4())
# # 将 UUID 记录在数据库中
# myuuid.insert_one({"uuid": uuid_value})
#
# csv_file = "data/Orange Juice.csv"  # 将此替换为你的 CSV 文件路径
# data = pd.read_csv(csv_file)
#
# # 将数据插入到 MongoDB 集合中
# mywrite.delete_many({})
# records = data.to_dict(orient='records')  # 将 DataFrame 转换为字典列表
# mywrite.insert_many(records)
# print("successfully write into database")
#
#
# data_from_mongodb = list(mywrite.find())
# # 将数据转换为 DataFrame
# data = pd.DataFrame(data_from_mongodb)
# # 假设你仍然需要筛选数据，你可以按照原来的方式进行：
# data = data[(data['store'] == store) & (data['brand'] == brand)]
# # 使用 MinMaxScaler 进行归一化处理
# scaler = MinMaxScaler(feature_range=(0, 1))
# data['Normalized_logmove'] = scaler.fit_transform(data['logmove'].values.reshape(-1, 1)).flatten()
#
# X, y, y_date, start_values = create_dataset(data[['Normalized_logmove']])
#
# X_train, X_valid, X_test, y_train, y_valid, y_test, time_train, time_valid, time_test = split_train_valid_test(
#         X,
#         y,data.week)
#
# print(X_train.shape, X_valid.shape, X_test.shape)
# print(y_train.shape, y_valid.shape, y_test.shape)
# print(time_train.shape, time_valid.shape, time_test.shape)

# mape_valid_lists = []
#
# for i in range(8):  # 循环迭代创建和训练八个不同的 LSTM 模型
#     model_function_name = f"create_LSTM_model{i}"
#     model = eval(model_function_name)()
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
#     model_name = f'models/LSTM_model_{i}.h5'  # 模型文件名中包含索引以区分不同的模型
#     es = EarlyStopping(monitor='mean_absolute_error', min_delta=0, patience=15, verbose=1, mode='auto')
#     mc = ModelCheckpoint(filepath=model_name, save_best_only=True)
#     callbacks = [es, mc]
#     print("training model ", i)
#     # 训练模型
#     fit = model.fit(
#             X_train, y_train,
#             batch_size=32,
#             epochs=200,
#             verbose=2,
#             validation_data=(X_valid, y_valid),
#             callbacks=callbacks)
#
#     #mape_train_list = fit.history['mean_absolute_error']  # 训练集的MAPE
#     mape_valid_list = fit.history['val_mean_absolute_error'][:100]  # 验证集的MAPE
#     mape_valid_lists.append(mape_valid_list)
#
#     mape_train,mape_valid,mape_test,pred_train,pred_valid,pred_test=evaluate_model(data, scaler, X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid,
#                        y_date_test, start_values, model)
#     myuuid.update_one({"uuid": uuid_value}, {"$set": {
#         "mape_train": mape_train,
#         "mape_valid": mape_valid,
#         "mape_test": mape_test
#     }})
#     myuuid.update_one({"uuid": uuid_value}, {"$set": {
#         "pred_train": pred_train.tolist(),
#         "pred_valid": pred_valid.tolist(),
#         "pred_test": pred_test.tolist()
#     }})
#
# plt.figure(figsize=(10, 6))
# for i in range(8):
#     #plt.plot(np.arange(1, len(mape_train_lists[i]) + 1), mape_train_lists[i], label=f'Training MAPE - Model {i}')
#     plt.plot(np.arange(1, len(mape_valid_lists[i]) + 1), mape_valid_lists[i], label=f'Model {i}')
#
# plt.title('MAPE vs. Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('MAPE')
# plt.legend()
# plt.grid(True)
# plt.show()
#


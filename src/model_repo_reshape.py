import os
from math import ceil, log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, \
    Dense, Dropout, Embedding, Flatten, Input, LSTM, Lambda, RepeatVector, \
    concatenate, dot
from tensorflow.keras.models import Sequential, Model, load_model
from keras.layers import Input, LSTM, RepeatVector, Dense, Flatten
from keras.models import Model


def create_DCNN_model(
        shape1,  # Lookback window of the lagged dynamic features
        shape2):
    n_outputs = 1 # How many timesteps of outputs to produce
    n_dilated_layers = 3
    kernel_size = 2
    n_filters = 3
    dropout_rate = 0.1
    # Sequential input for dynamic features
    seq_in = Input(shape=(shape1, shape2))
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
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])

    return model


def create_LSTM_model1(shape1,shape2):
    n_outputs = 1
    latent_dim = 128
    # latent_dim = 8, 41
    encoder_inputs = Input(shape=(shape1, shape2))
    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, shape1, shape2),
                   stateful=False,
                   return_sequences=False,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    decoder_inputs = RepeatVector(n_outputs)(state_h)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, shape2 + 0),
                     stateful=False,
                     return_sequences=True,
                     return_state=True,
                     dropout=0.2,
                     recurrent_dropout=0.2)

    decoder_outputs, _, _ = decoder_1(
        decoder_inputs, initial_state=[state_h, state_c])
    decoder_outputs = Flatten()(decoder_outputs)

    d = Dense(16, activation="relu")(decoder_outputs)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])

    return model

def create_LSTM_model2(shape1,shape2):
    n_outputs = 1
    latent_dim = 128
    encoder_inputs = Input(shape=(shape1, shape2))
    encoder = LSTM(latent_dim,
                   batch_input_shape=(1, shape1, shape2 + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = RepeatVector(n_outputs)(state_h)

    decoder_1 = LSTM(latent_dim,
                     batch_input_shape=(1, n_outputs, shape2 + 0),
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

    d = Dense(16, activation="relu")(decoder_outputs)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])

    return model


def create_LSTM_model3(shape1,shape2):
    n_outputs = 1
    latent_dim = 60
    # 1层LSTM decoder, with attention
    # latent_dim=8 38
    encoder_inputs = Input(shape=(shape1, shape2))

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, shape1, shape2 + 1),
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
                     batch_input_shape=(1, n_outputs, shape2 + 0),
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

    d = Dense(16, activation="relu")(decoder_outputs)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs, outputs=output)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])
    return model


def create_LSTM_model4(shaoe1,shape2):
    n_outputs=1
    latent_dim=60
    encoder_inputs = Input(shape=(shaoe1, shape2))
    k = 4

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, shaoe1, shape2 + 1),
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
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])

    return model


def create_LSTM_model5(shape1,shape2):
    n_outputs = 1
    latent_dim = 60
    encoder_inputs = Input(shape=(shape1, shape2))
    k = 4

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, shape1, shape2 + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_outputs = tf.reshape(
        encoder_outputs, [-1, latent_dim, shape1])  # latent_dim * seq_len

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
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])
    return model


def create_LSTM_model6(shape1,shape2):
    n_outputs = 1
    latent_dim = 60
    encoder_inputs = Input(shape=(shape1, shape2))
    k = 4

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, shape1, shape2 + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    e = encoder_outputs[:, :-1, :]
    e = tf.reshape(e, [-1, latent_dim, shape1 - 1])  # latent_dim * seq_len
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
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])
    return model


def create_LSTM_model7(shape1,shape2):
    n_outputs=1
    latent_dim=60
    encoder_inputs = Input(shape=(shape1, shape2))
    k = 4

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, shape1, shape2 + 1),
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
    decoder_inputs = tf.reshape(decoder_inputs, [-1, latent_dim * 2, shape1])
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

    d = Dense(16, activation="relu")(decoder_outputs)
    output = Dense(n_outputs, activation="linear")(d)

    model = Model(inputs=encoder_inputs ,outputs=output)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'mape'])

    return model
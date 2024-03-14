#!/usr/bin/env python3
# encoding: utf-8

"""Houses the definitions of all candidate models used for the recs."""

# pylint: disable
# Do not bother to lint or clean up this mess

import os
from math import ceil, log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, \
    Dense, Dropout, Embedding, Flatten, Input, LSTM, Lambda, RepeatVector, \
    concatenate, dot
from tensorflow.keras.models import Model


def create_DCNN_model(
        max_cat_id,  # Size of the categorical feature vocabularies
        seq_len,  # Lookback window of the lagged dynamic features
        n_dyn_fea=1,  # Number of dynamic features
        n_outputs=1,  # How many timesteps of outputs to produce
        n_dilated_layers=3,
        kernel_size=2,
        n_filters=3,
        dropout_rate=0.1):
    """Create a Dilated CNN model.
    Args:
        seq_len (int): Input sequence length
        n_dyn_fea (int): Number of dynamic features
        n_outputs (int): Number of outputs of the network
        kernel_size (int): Kernel size of each convolutional layer
        n_filters (int): Number of filters in each convolutional layer
        dropout_rate (float): Dropout rate in the network
        max_cat_id (list[int]): Each entry in the list represents the
        maximum value of the ID of a specific categorical variable.
    Returns:
        object: Keras Model object
    """
    # Sequential input for dynamic features
    seq_in = Input(shape=(seq_len, n_dyn_fea))

    # Categorical input
    n_cat_fea = len(max_cat_id)
    cat_fea_in = Input(shape=(n_cat_fea,), dtype="uint8")
    cat_flatten = []
    for i, m in enumerate(max_cat_id):
        cat_fea = Lambda(lambda x, i: x[:, i, None], arguments={
            "i": i})(cat_fea_in)
        cat_fea_embed = Embedding(
            m + 1, ceil(log(m + 1)), input_length=1)(cat_fea)
        cat_flatten.append(Flatten()(cat_fea_embed))

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

    x = concatenate([conv_out] + cat_flatten)
    # Concatenate with categorical features

    x = Dense(16, activation="relu")(x)
    output = Dense(n_outputs, activation="linear")(x)

    # Define model interface
    inputs = [seq_in, cat_fea_in]

    model = Model(inputs=inputs, outputs=output)

    return model


def create_LSTM_model(
        max_cat_id,
        seq_len=15,

        n_dyn_fea=6,
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


def create_LSTM_model2(max_cat_id,
                       seq_len=15,
                       n_dyn_fea=6,
                       n_outputs=2,
                       latent_dim=128
                       ):
    # 两层decoder
    # latent_dim = 8, 40
    encoder_inputs = Input(shape=(seq_len, n_dyn_fea))

    encoder = LSTM(latent_dim,  # latent dim 表示输出，h和c的维度
                   batch_input_shape=(1, seq_len, n_dyn_fea + 1),
                   stateful=False,
                   return_sequences=True,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
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
    decoder_2 = LSTM(120,
                     stateful=False,
                     return_sequences=True,
                     return_state=True,
                     dropout=0.2,
                     recurrent_dropout=0.2)

    decoder_outputs, _, _ = decoder_2(
        decoder_1(decoder_inputs, initial_state=encoder_states))
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


def create_LSTM_model3(max_cat_id,
                       seq_len=15,
                       n_dyn_fea=6,
                       n_outputs=2,
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
    print(output)

    model = Model(inputs=[encoder_inputs, cat_fea_in], outputs=output)
    return model


def create_LSTM_model4(max_cat_id,
                       seq_len=15,
                       n_dyn_fea=6,
                       n_outputs=2,
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

    # m * latent_dim, m * latent_dim -> m * m
    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)

    # m * m, m * latent_dim -> m * latent_dim
    context = dot([attention, encoder_outputs], axes=[2, 1])
    context = BatchNormalization(momentum=0.6)(context)

    decoder_outputs = concatenate([context, decoder_outputs])
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


def create_LSTM_model5(max_cat_id,
                       seq_len=15,
                       n_dyn_fea=6,
                       n_outputs=2,
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


def create_LSTM_model6(max_cat_id,
                       seq_len=15,
                       n_dyn_fea=6,
                       n_outputs=2,
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


def create_LSTM_model7(max_cat_id,
                       seq_len=15,
                       n_dyn_fea=6,
                       n_outputs=2,
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

    # state_h = BatchNormalization(momentum=0.6)(state_h)

    # decoder_inputs = RepeatVector(n_outputs)(state_h)  # n_outputs *
    # latent_dim
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

import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from utils import attention_block
from utils import Time2Vector

#--------------------------------------------------------------------------------------------
#           Hyperparameter setting (Demo)
#--------------------------------------------------------------------------------------------

INPUT_DIMS = 3
TIME_STEPS = 20
cnn_filter = 64
cnn_kernel  = 2
lstm_units = 64
dropout = 0.1


def attention_model():
    in_seq = Input(shape=(TIME_STEPS, INPUT_DIMS))
    time_embedding = Time2Vector(TIME_STEPS)
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = Conv1D(filters = cnn_filter, kernel_size = cnn_kernel, activation = 'selu', padding = 'causal')(x)

    # For GPU you can use CuDNNLSTM
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(dropout)(lstm_out)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_out)
    lstm_out = Dropout(dropout)(lstm_out)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_out)
    lstm_out = Dropout(dropout)(lstm_out)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_out)
    lstm_out = Dropout(dropout)(lstm_out)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_out)
    lstm_out = Dropout(dropout)(lstm_out)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_out)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1)(attention_mul)
    model = Model(inputs=in_seq, outputs=output)
    return model


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import NormalizeMult, create_dataset, r2_keras
from keras.optimizers import Adam, SGD, Nadam
from network import attention_model
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Set to -1 if CPU should be used CPU = -1 , GPU = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
cpus = tf.config.experimental.list_physical_devices('CPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
elif cpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        logical_cpus= tf.config.experimental.list_logical_devices('CPU')
        print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

path = ""
df = pd.read_csv(path,index_col=[0])

train_size = int(len(df)*0.8) #training + validation / testing
train = df.iloc[:train_size,:]
test = df.iloc[train_size:,:]



#--------------------------------------------------------------------------------------------
#           Hyperparameter setting (Demo)
#--------------------------------------------------------------------------------------------


INPUT_DIMS = 3
TIME_STEPS = 20
lstm_units = 64


epoch = 2
batch_size = 64
lr = 0.001
validation_split = 0.1
#--------------------------------------------------------------------------------------------
#           Normalized
#--------------------------------------------------------------------------------------------

train,normalize = NormalizeMult(train)
WP_data = train[:,0].reshape(len(train),1)

train_X, _ = create_dataset(train,TIME_STEPS)
_ , train_Y = create_dataset(WP_data,TIME_STEPS)


#--------------------------------------------------------------------------------------------
#           Model training (version without grid search)
#--------------------------------------------------------------------------------------------


Attention_based_model = attention_model()
adam = Adam(lr=lr)
Attention_based_model.summary()
Attention_based_model.compile(loss='mse',
                              optimizer=adam,
                              metrics=['mae',r2_keras])

# fit the network
history =  Attention_based_model.fit([train_X],
                                     train_Y,
                                     epochs=epoch,
                                     batch_size=batch_size,
                                     validation_split=validation_split)

#--------------------------------------------------------------------------------------------
#           Plot
#--------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# summarize history for R^2
fig_r2 = plt.figure(figsize=(6, 5))
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model r^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# summarize history for MAE
fig_mae = plt.figure(figsize=(10, 10))
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for Loss
fig_loss = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#--------------------------------------------------------------------------------------------
#           test
#--------------------------------------------------------------------------------------------

test,normalize = NormalizeMult(test)
WP_test = test[:,0].reshape(len(test),1)

test_X, _ = create_dataset(test,TIME_STEPS)
_ , test_Y = create_dataset(WP_test,TIME_STEPS)


# Prediction
scores_test = Attention_based_model.evaluate([test_X], test_Y, verbose=2)
results = Attention_based_model.predict([test_X])

#--------------------------------------------------------------------------------------------
#           result plot
#--------------------------------------------------------------------------------------------
fig_result = plt.figure(figsize=(12, 5))
plt.plot(results,alpha=0.8)
plt.plot(test_Y,alpha=0.8)
plt.title('real vs pred')
plt.ylabel('value')
plt.xlabel('sample')
plt.legend(['pred', 'real'], loc='upper left')
plt.grid()
plt.show()


#--------------------------------------------------------------------------------------------
#           Metrics
#--------------------------------------------------------------------------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer,r2_score
import math


def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return rmse

def final_result(test, results):
    print(round(mean_squared_error(test, results)*1000,5))
    print(round(mean_absolute_error(test, results)*100,5))
    print(round(return_rmse(test, results)*100,5))

final_result(test_Y, results)
print(round(r2_score(test_Y, results),5))
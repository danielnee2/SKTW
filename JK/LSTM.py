import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
# from utility import *

# homedir = get_homedir()
# FIPS_mapping, FIPS_full = get_FIPS(reduced=True)
quantileList = np.array(range(1, 10, 1)) * 0.1

def get_TRAIN_SPLIT(history_size, target_size, total_size, split_ratio=0.2):
    """
    Return the splitting date (=number of training dates) for KFold splitting.
    Dates are assumed 1-step, 0-based.

    Parameters:
      history_size: int
        Size of history window.
      target_size: int
        Size of target window.
      total_size: int
        Total number of sample dates.
      split_ratio: int (default=0.2)
        K-fold ratio of train-validation split.
    
    Return:
      TRAIN_SPLIT: int
        Splitting date = number of training dates.
    """
    assert total_size>=2*history_size+target_size-1+(2/split_ratio), 'History and Target sizes are too large.'

    return int((1-split_ratio)*(total_size-target_size+1)-(1-2*split_ratio)*history_size)+1

def train_val_split(dataList, target_idx, history_size, target_size, total_size=None, split_ratio=0.2, step_size=1):
    """
    """
    if total_size is None:
        total_size = len(dataList[0])
    TRAIN_SPLIT = get_TRAIN_SPLIT(history_size, target_size, total_size, split_ratio=split_ratio)

    X_train, y_train = [], []
    X_val, y_val = [], []

    for data in dataList:
        for i in range(history_size, TRAIN_SPLIT, step_size):
            X_train.append(data[i-history_size:i, :])
            y_train.append(data[i:i+target_size, target_idx])
        for i in range(TRAIN_SPLIT+history_size, total_size-target_size+1, step_size):
            X_val.append(data[i-history_size:i, :])
            y_val.append(data[i:i+target_size, target_idx])

    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val)

def get_StandardScaler(X_train, target_idx=None):
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).astype(np.float32))

    if target_idx:
        mu, sigma = scaler.scale_[target_idx], scaler.mean_[target_idx]
    else:
        mu, sigma = None, None

    return scaler, mu, sigma

def normalizer(scaler, X, y, target_idx):
    mu, sigma = scaler.mean_[target_idx], scaler.scale_[target_idx]

    X = np.asarray(np.vsplit(scaler.transform(np.vstack(X)), len(X)))
    y = (y - mu) / sigma
    
    return X, y

def load_Dataset(X_train, y_train, X_val, y_val, BATCH_SIZE=32, BUFFER_SIZE=10000):
    """
    Popular BATCH_SIZE: 32, 64, 128
    Oftentimes smaller BATCH_SIZE perform better
    """
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()
    
    return train_data, val_data

def quantileLoss(quantile, y_p, y):
    """
    Costum loss function for quantile forecast models.
    Intended usage:
    >>> loss=lambda y_p, y: quantileLoss(quantile, y_p, y)
    in compile step.

    Parameters:
      quantile: float in [0,1]
        Quantile number
    """
    e = y_p - y
    return tf.keras.backend.mean(tf.keras.backend.maximum(quantile*e, (quantile-1)*e))

def LSTM_fit(train_data, val_data, lr=0.001, NUM_CELLS=128, EPOCHS=10, EVALUATION_INTERVAL=200, monitor=False):
    history_size = train_data.element_spec[0].shape[1]
    feature_size = train_data.element_spec[0].shape[2]
    target_size = train_data.element_spec[1].shape[1]
    
    model_qntl = []
    history_qntl =[]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    for quantile in quantileList:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(NUM_CELLS, return_sequences=True, input_shape=(history_size, feature_size)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LSTM(round(NUM_CELLS/2), activation='relu', return_sequences=True))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LSTM(round(NUM_CELLS/4), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(target_size))
        
        model.compile(optimizer=optimizer, loss=lambda y_p, y: quantileLoss(quantile, y_p, y))
        history = model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data, validation_steps=50)
        model_qntl.append(model)
        history_qntl.append(history)

    if monitor:
        return model_qntl, history_qntl
    else:
        return model_qntl

def predict_future(model_qntl, dataList, scaler, target_idx, FIPS=None, date_ed=None):
    mu, sigma = scaler.mean_[target_idx], scaler.scale_[target_idx]
    history_size = model_qntl[0].input.shape[1]
    target_size = model_qntl[0].output.shape[1]

    X_future = [data[-history_size:, :] for data in dataList]
    X_future = np.asarray(X_future)
    X_future = np.asarray(np.vsplit(scaler.transform(np.vstack(X_future)), len(X_future)))

    prediction_future = []
    for i in range(len(model_qntl)):
        prediction_future.append(sigma*model_qntl[i].predict(X_future)+mu)

    if (FIPS is None) or (date_ed is None):
        return np.asarray(prediction_future)
    else:
        df_future = []
        for i, fips in enumerate(FIPS):
            for j in range(target_size):
                df_future.append([date_ed+pd.Timedelta(days=1+j), fips]+prediction_future[:,i,j].tolist())

        return pd.DataFrame(df_future, columns=['date', 'fips']+list(range(10, 100, 10)))

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

def plot_prediction(model_qntl, val_data, scaler, target_idx, num=3):
    mu, sigma = scaler.mean_[target_idx], scaler.scale_[target_idx]
    
    for x, y in val_data.take(num):
        X_unnm = scaler.inverse_transform(x)
        y_unnm = sigma * y + mu
        
        plt.figure(figsize=(12,6))
        plt.plot(list(range(-len(X_unnm[0]), 0)), np.array(X_unnm[0][:, target_idx]), label='History')
        plt.plot(np.arange(len(y_unnm[0])), np.array(y_unnm[0]), 'bo', label='True Future')
        for i in [0, 4, 8]:
            prediction = model_qntl[i].predict(X_unnm)[0]
            prediction = sigma * prediction + mu
            plt.plot(np.arange(len(y_unnm[0])), np.array(prediction), 'o', label=f'Predicted Future, qntl={10*(i+1)}')
        plt.legend(loc='upper left')
        plt.show()
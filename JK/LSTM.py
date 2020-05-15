import platform
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
# from utility import *

# homedir = get_homedir()
# FIPS_mapping, FIPS_full = get_FIPS(reduced=True)
quantileList = np.linspace(0.1, 0.9, 9)
if platform.system()=='Linux': ### In a session
    import matplotlib
    matplotlib.use('Agg')

class LRFinder(tf.keras.callbacks.Callback):
    """Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the plot method.
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 10 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self, path=None):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)
        if path is None:
            fig.show()
        else:
            fig.savefig(path)

class ConditionalRNN(tf.keras.layers.Layer):
    
    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, units, cell=tf.keras.layers.LSTMCell, *args,
                 **kwargs):
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param cell: string, cell class or object (pre-instantiated). In the case of string, 'GRU',
        'LSTM' and 'RNN' are supported.
        :param args: Any parameters of the tf.keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        super().__init__()
        self.units = units
        self.final_states = None
        self.init_state = None
        if isinstance(cell, str):
            if cell.upper() == 'GRU':
                cell = tf.keras.layers.GRUCell
            elif cell.upper() == 'LSTM':
                cell = tf.keras.layers.LSTMCell
            elif cell.upper() == 'RNN':
                cell = tf.keras.layers.SimpleRNNCell
            else:
                raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        self._cell = cell if hasattr(cell, 'units') else cell(units=units)
        self.rnn = tf.keras.layers.RNN(cell=self._cell, *args, **kwargs)

        # single cond
        self.cond_to_init_state_dense_1 = tf.keras.layers.Dense(units=self.units, *args, **kwargs)

        # multi cond
        max_num_conditions = 10
        self.multi_cond_to_init_state_dense = []
        for i in range(max_num_conditions):
            self.multi_cond_to_init_state_dense.append(tf.keras.layers.Dense(units=self.units, *args, **kwargs))
        self.multi_cond_p = tf.keras.layers.Dense(1, activation=None, use_bias=True, *args, **kwargs)

    def _standardize_condition(self, initial_cond):
        initial_cond_shape = initial_cond.shape
        if len(initial_cond_shape) == 2:
            initial_cond = tf.expand_dims(initial_cond, axis=0)
        first_cond_dim = initial_cond.shape[0]
        if isinstance(self._cell, tf.keras.layers.LSTMCell):
            if first_cond_dim == 1:
                initial_cond = tf.tile(initial_cond, [2, 1, 1])
            elif first_cond_dim != 2:
                raise Exception('Initial cond should have shape: [2, batch_size, hidden_size]\n'
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        elif isinstance(self._cell, tf.keras.layers.GRUCell) or isinstance(self._cell, tf.keras.layers.SimpleRNNCell):
            if first_cond_dim != 1:
                raise Exception('Initial cond should have shape: [1, batch_size, hidden_size]\n'
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        else:
            raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        return initial_cond

    def __call__(self, inputs, *args, **kwargs):
        """
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        assert isinstance(inputs, list) and len(inputs) >= 2, f"{inputs}"
        x = inputs[0]
        cond = inputs[1:]
        if len(cond) > 1:  # multiple conditions.
            init_state_list = []
            for ii, c in enumerate(cond):
                init_state_list.append(self.multi_cond_to_init_state_dense[ii](self._standardize_condition(c)))
            multi_cond_state = self.multi_cond_p(tf.stack(init_state_list, axis=-1))
            multi_cond_state = tf.squeeze(multi_cond_state, axis=-1)
            self.init_state = tf.unstack(multi_cond_state, axis=0)
        else:
            cond = self._standardize_condition(cond[0])
            if cond is not None:
                self.init_state = self.cond_to_init_state_dense_1(cond)
                self.init_state = tf.unstack(self.init_state, axis=0)
        out = self.rnn(x, initial_state=self.init_state, *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states
        else:
            return out

class SingleLayerConditionalRNN(tf.keras.Model):
    def __init__(self, NUM_CELLS, target_size, **kwargs):
        super().__init__()
        self.layer1 = ConditionalRNN(NUM_CELLS, cell='LSTM', **kwargs)
        self.out = tf.keras.layers.Dense(target_size, **kwargs)

    def call(self, inputs, **kwargs):
        o = self.layer1(inputs)
        o = self.out(o)
        return o

def _get_TRAIN_SPLIT(history_size, target_size, total_size, split_ratio=0.2):
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

def train_val_split(data_ts, data_ctg, target_idx, history_size, target_size, split_ratio=0.2, step_size=1):
    """
    """
    total_size = len(data_ts[0])
    TRAIN_SPLIT = _get_TRAIN_SPLIT(history_size, target_size, total_size, split_ratio=split_ratio)

    assert len(data_ts)==len(data_ctg), "Length of timeseries and categorical data do not match."

    X_train, y_train = [], []
    X_val, y_val = [], []
    Ctg_train, Ctg_val = [], []

    for fips in range(len(data_ts)):
        for i in range(history_size, TRAIN_SPLIT, step_size):
            X_train.append(data_ts[fips][i-history_size:i, :])
            y_train.append(data_ts[fips][i:i+target_size, target_idx])
            Ctg_train.append(data_ctg[fips])
        for i in range(TRAIN_SPLIT+history_size, total_size-target_size+1, step_size):
            X_val.append(data_ts[fips][i-history_size:i, :])
            y_val.append(data_ts[fips][i:i+target_size, target_idx])
            Ctg_val.append(data_ctg[fips])

    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val), np.asarray(Ctg_train), np.asarray(Ctg_val)

def get_StandardScaler(X_train, X_ctg):
    scaler_ts, scaler_ctg = StandardScaler(), StandardScaler()
    scaler_ts.fit(np.vstack(X_train).astype(np.float32))
    scaler_ctg.fit(X_ctg.astype(np.float32))

    return scaler_ts, scaler_ctg

def normalizer(scaler, X, y=None, target_idx=None):
    if target_idx is None:
        X = scaler.transform(X)
        return X
    else:
        mu, sigma = scaler.mean_[target_idx], scaler.scale_[target_idx]

        X = np.asarray(np.vsplit(scaler.transform(np.vstack(X)), len(X)))
        y = (y - mu) / sigma
    
        return X, y

def load_Dataset(X_train, C_train, y_train, X_val, C_val, y_val, BATCH_SIZE=32, BUFFER_SIZE=10000):
    """
    Popular BATCH_SIZE: 32, 64, 128
    Oftentimes smaller BATCH_SIZE perform better
    """
    X_tr_data = tf.data.Dataset.from_tensor_slices(X_train)
    C_tr_data = tf.data.Dataset.from_tensor_slices(C_train)
    y_tr_data = tf.data.Dataset.from_tensor_slices(y_train)
    train_data = tf.data.Dataset.zip(((X_tr_data, C_tr_data), y_tr_data))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    X_v_data = tf.data.Dataset.from_tensor_slices(X_val)
    C_v_data = tf.data.Dataset.from_tensor_slices(C_val)
    y_v_data = tf.data.Dataset.from_tensor_slices(y_val)
    val_data = tf.data.Dataset.zip(((X_v_data, C_v_data), y_v_data))
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
    return tf.math.reduce_mean(tf.math.maximum(quantile*e, (quantile-1)*e))

def LSTM_fit(train_data, val_data, lr=0.001, NUM_CELLS=128, EPOCHS=10, dp=0.2, monitor=False, **kwargs):
    target_size = train_data.element_spec[1].shape[1]
    
    model_qntl = [SingleLayerConditionalRNN(NUM_CELLS, target_size) for _ in range(len(quantileList))]
    history_qntl =[]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for i in range(len(quantileList)):
        model_qntl[i].compile(optimizer=optimizer, loss=lambda y_p, y: quantileLoss(quantileList[i], y_p, y))
        print(f'Quantile={10*(i+1)} is trained')
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        history = model_qntl[i].fit(train_data, epochs=EPOCHS, steps_per_epoch=200, validation_data=val_data,
                                    validation_steps=50, callbacks=[earlystop], shuffle=True, **kwargs)
        history_qntl.append(history)

    if monitor:
        return model_qntl, history_qntl
    else:
        return model_qntl

def LSTM_finder(train_data, val_data, lr=0.001, NUM_CELLS=128, EPOCHS=10, dp=0.2, **kwargs):
    target_size = train_data.element_spec[1].shape[1]
    
    model_qntl = [SingleLayerConditionalRNN(NUM_CELLS, target_size) for _ in range(len(quantileList))]
    history_qntl =[]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    lr_finder = [LRFinder() for _ in range(len(quantileList))]

    for i in range(len(quantileList)):
        model_qntl[i].compile(optimizer=optimizer, loss=lambda y_p, y: quantileLoss(quantileList[i], y_p, y))
        print(f'Quantile={10*(i+1)} is trained')
        history = model_qntl[i].fit(train_data, epochs=EPOCHS, steps_per_epoch=200, validation_data=val_data, validation_steps=50, callbacks=[lr_finder[i]])

    return lr_finder

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
    prediction_future = np.asarray(prediction_future)

    if (FIPS is None) or (date_ed is None):
        return np.asarray(prediction_future)
    else:
        print('Saving future prediction.')
        df_future = []
        for i, fips in enumerate(FIPS):
            for j in range(target_size):
                df_future.append([date_ed+pd.Timedelta(days=1+j), fips]+prediction_future[:,i,j].tolist())

        return pd.DataFrame(df_future, columns=['date', 'fips']+list(range(10, 100, 10)))

def plot_train_history(history, title='Untitled', path=None):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

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

# for i, quantile in enumerate(quantileList):
#     model = condLSTM(NUM_CELLS, NUM_DAYS_OUT)
#     model.call([trainingData_time, trainingData_static])
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     #model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])
#     model.compile(optimizer=optimizer, loss=lambda y_p, y: quantileLoss(quantile, y_p, y))
#     model.fit(x=[trainingData_time, trainingData_static], y=trainingAns_time,
#               epochs=epochList[i])
#     modelList.append(model)
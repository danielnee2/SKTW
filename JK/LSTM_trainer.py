import os
from time import time, process_time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from utility import *
from LSTM import *

homedir = get_homedir()

TODAY = '0519' # date of today, in the form of string. It signifies which folder in JK/preprocessing to be trained.
date_ed = pd.Timestamp('2020-05-16') # end date to be included in the training. in the form of pandas Timestamp. 
                                     # should be the same as an output of DataCleaner.py.

PATH_PREP = f"{homedir}/JK/preprocessing/{TODAY}"
PATH = f"{homedir}/JK/prediction/{TODAY}"
split_ratio = 0.1
QUANTILE = list(quantileList)
history_size = 7
target_size = 14
step_size = 1
NUM_CELLS = 128
lr = 0.001
dp = 0.2
EPOCHS = 100
#######################################################################################

with open(PATH_PREP+f'/FIPS.txt', 'r') as f:
    FIPS_total = eval(f.read())

data_ctg = np.load(PATH_PREP+f'/data_ctg.npy', allow_pickle=True)
print(f'Categorical data of shape {data_ctg.shape} is loaded.')
data_ts = np.load(PATH_PREP+f'/data_ts.npy', allow_pickle=True)
print(f'Timeseries data of shape {data_ts.shape} is loaded.')

with open(PATH_PREP+f'/columns_ctg.txt', 'r') as f:
    columns_ctg = eval(f.read())
with open(PATH_PREP+f'/columns_ts.txt', 'r') as f:
    columns_ts = eval(f.read())
print(f'# of features = {len(columns_ctg)+len(columns_ts)}')

target_idx = 1 #columns.index('deaths')

try:
    os.mkdir(PATH)
except OSError as error:
    print(error)

X_train, y_train, X_val, y_val, C_train, C_val = train_val_split(data_ts, data_ctg, target_idx, history_size, target_size, split_ratio=split_ratio, step_size=step_size)

scaler_ts, scaler_ctg = get_StandardScaler(X_train, C_train)

X_train, y_train = normalizer(scaler_ts, X_train, y_train, target_idx)
X_val, y_val = normalizer(scaler_ts, X_val, y_val, target_idx)
C_train, C_val = normalizer(scaler_ctg, C_train), normalizer(scaler_ctg, C_val)

train_data, val_data = load_Dataset(X_train, C_train, y_train, X_val, C_val, y_val)

lr_finder = LSTM_finder(train_data, val_data, lr=lr, NUM_CELLS=NUM_CELLS, EPOCHS=EPOCHS, dp=dp)
# model_qntl, history_qntl = LSTM_fit(train_data, val_data, lr=lr, NUM_CELLS=NUM_CELLS, EPOCHS=EPOCHS, dp=dp, monitor=True, verbose=2)

for i in range(len(QUANTILE)):
    FILEPATH = f"/LSTM_qntl={10*(i+1)}"
    lr_finder[i].plot(PATH+FILEPATH+'.png')
    # plot_train_history(history_qntl[i], title=f'History size={history_size}, dropout={dp}', path=PATH+FILEPATH+'_history.png')
    # model_qntl[i].save(PATH+FILEPATH)
    # np.save(PATH+FILEPATH+'.npy', np.vstack((LOSS, VAL_LOSS)).astype(np.float32))

# df_future = predict_future(model_qntl, data_ts, data_ctg, scaler_ts, scaler_ctg, history_size, target_idx, FIPS=FIPS_total, date_ed=date_ed)
# df_future.to_csv(PATH+f'/LSTM_{TODAY}.csv', index=False)
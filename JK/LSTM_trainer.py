import os
from time import time, process_time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from utility import *
from LSTM import *

homedir = get_homedir()

TODAY = '0514'
date_ed = pd.Timestamp('2020-05-08')

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
EPOCHS = 5
#######################################################################################

with open(PATH_PREP+f'/FIPS.txt', 'r') as f:
    FIPS_total = eval(f.read())

data_ctg = np.load(PATH_PREP+f'/data_ctg.npy', allow_pickle=True)
print(f'Categorical data of shape {data_ctg.shape} is loaded.')
data_ts = np.load(PATH_PREP+f'/data_ts.npy', allow_pickle=True)
print(f'Timeseries data of shape {data_ts.shape} is loaded.')

with open(PATH_PREP+f'/columns.txt', 'r') as f:
    columns = eval(f.read())
print(f'# of features = {len(columns)}')

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

# history_size = train_data.element_spec[0].shape[1]
# feature_size = train_data.element_spec[0].shape[2]
# target_size = train_data.element_spec[1].shape[1]

# lr_finder = LSTM_finder(train_data, val_data, lr=lr, NUM_CELLS=NUM_CELLS, EPOCHS=EPOCHS, dp=dp)
model_qntl, history_qntl = LSTM_fit(train_data, val_data, lr=lr, NUM_CELLS=NUM_CELLS, EPOCHS=EPOCHS, dp=dp, monitor=True)

for i in range(len(QUANTILE)):
    FILEPATH = f"/LR_finder_qntl={10*(i+1)}"
    # lr_finder[i].plot(PATH+FILEPATH+'.png')
    plot_train_history(history_qntl[i], title=f'History size={history_size}, dropout={dp}', path=PATH+FILEPATH+'_history.png')
    np.save(PATH+FILEPATH+'.npy', np.vstack((LOSS, VAL_LOSS)).astype(np.float32))

# df_future = predict_future(model_qntl, dataList[c], scaler, target_idx, FIPS=FIPS_cluster[c], date_ed=date_ed)
# df_future.to_csv(PATH+f'/LSTM_class={c}_{TODAY}.csv', index=False)
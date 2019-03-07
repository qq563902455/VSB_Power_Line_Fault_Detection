import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyarrow.parquet as pq
import os
import random
import gc

import torch
import torch.nn as nn
import torch.utils.data

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold;



from specialTools.rnn_model import LSTM_softmax
from specialTools.threePhasesNNModel import threePhasesModel


train = pd.read_csv('./processedData/train_DL.csv')
test = pd.read_csv('./processedData/test_DL.csv')

train_features = np.load('./processedData/train_DL_features.npy')
test_features = np.load('./processedData/test_DL_features.npy')

train['features'] = list(train_features)
test['features'] = list(test_features)

del train_features
del test_features
gc.collect()

def mcc_metric(y_true, y_pred_proba):

    best_score = 0
    best_threshold = 0

    for threshold in [val/100 for val in range(100)]:

        y_pred = y_pred_proba > threshold

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        fp = np.sum((y_true == 1) & (y_pred == 0))
        fn = np.sum((y_true == 0) & (y_pred == 1))

        if tp == 0 or tn == 0: continue

        score = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

        if score > best_score:
            best_score = score
            best_threshold = threshold


    return best_score, best_threshold

seq_len = train.features[0].shape[0]
features_dims = train.features[0].shape[1]

modellist = [LSTM_softmax, LSTM_softmax, LSTM_softmax]
modelParamList = [
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,
            'lstm_out_dim': 100,
            'lstm_layers': 2,
        },
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,
            'lstm_layers': 2,
            'lstm_out_dim': 100,

        },
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,
            'lstm_out_dim': 100,
        }
]
phaseList = [0, 1, 2]
train_epochs_list = [100, 100, 100]
batch_size_list = [32, 32, 32]

fold_num = 5
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=10)

model = threePhasesModel(modellist, modelParamList, phaseList)
model.fit(train, kfold, train_epochs_list, batch_size_list, mcc_metric)

score, threshold = mcc_metric(model.train_total_target, model.train_total_preds)
print('total score:\t', score, 'threshold:\t', threshold)

answer = model.predict_proba(test)

answer.target = (answer.target > threshold).astype(int)
answer.to_csv('./submit.csv', index=False)

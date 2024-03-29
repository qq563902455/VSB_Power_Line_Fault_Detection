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
from specialTools.rnn_model import LSTM_selfAttention_softmax

from specialTools.threePhasesNNModel import threePhasesModel
from specialTools.threePhasesNNModel import mcc_metric


train = pd.read_csv('./processedData/train_DL.csv')
test = pd.read_csv('./processedData/test_DL.csv')

train_features = np.load('./processedData/train_DL_features.npy')
test_features = np.load('./processedData/test_DL_features.npy')


train['features'] = list(train_features)
test['features'] = list(test_features)

del train_features
del test_features
gc.collect()


seq_len = train.features[0].shape[0]
features_dims = train.features[0].shape[1]


# model 1

modellist = [LSTM_selfAttention_softmax, LSTM_selfAttention_softmax, LSTM_selfAttention_softmax]
modelParamList = [
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,

            'lstm_layers': 2,
            'lstm_out_dim': 100,

            'selfAttention_dim': 200,
            'linearReduction_dim': 64,

            'env': 'phase_0',
        },
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,

            'lstm_layers': 2,
            'lstm_out_dim': 100,

            'selfAttention_dim': 200,
            'linearReduction_dim': 64,

            'env': 'phase_1',
        },
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,

            'lstm_layers': 2,
            'lstm_out_dim': 100,

            'selfAttention_dim': 200,
            'linearReduction_dim': 64,

            'env': 'phase_2',
        }
]
phaseList = [0, 1, 2]
train_epochs_list = [100, 100, 100]
batch_size_list = [32, 32, 32]

fold_num = 5
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=10)

model_1 = threePhasesModel(modellist, modelParamList, phaseList)
model_1.fit(train, kfold, train_epochs_list, batch_size_list)



# model 2

modellist = [LSTM_softmax, LSTM_softmax, LSTM_softmax]
modelParamList = [
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,

            'lstm_layers': 2,
            'lstm_out_dim': 100,

            'linearReduction_dim': 64,

            'env': 'phase_0',
        },
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,

            'lstm_layers': 2,
            'lstm_out_dim': 100,

            'linearReduction_dim': 64,

            'env': 'phase_1',
        },
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'learning_rate': 0.0003,

            'lstm_layers': 2,
            'lstm_out_dim': 100,

            'linearReduction_dim': 64,

            'env': 'phase_2',
        }
]
phaseList = [0, 1, 2]
train_epochs_list = [100, 100, 100]
batch_size_list = [32, 32, 32]

model_2 = threePhasesModel(modellist, modelParamList, phaseList)
model_2.fit(train, kfold, train_epochs_list, batch_size_list)





score, threshold = mcc_metric(model_1.train_total_target, model_1.train_total_preds)
print('model_1 score:\t', score, 'threshold:\t', threshold)



score, threshold = mcc_metric(model_2.train_total_target, model_2.train_total_preds)
print('model_2 score:\t', score, 'threshold:\t', threshold)


score, threshold = mcc_metric(model_1.train_total_target, 0.9*model_1.train_total_preds+0.1*model_2.train_total_preds)
print('total score:\t', score, 'threshold:\t', threshold)


answer_1 = model_1.predict_proba(test)
answer_2 = model_2.predict_proba(test)

answer= answer_1.copy()

answer.target = 0.9*answer_1.target + 0.1*answer_2.target

answer.target = (answer.target > threshold).astype(int)
answer.to_csv('./submit.csv', index=False)

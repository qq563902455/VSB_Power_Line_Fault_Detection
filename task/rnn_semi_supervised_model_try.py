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
from specialTools.rnn_model import semi_superviesd_LSTM_selfAttention
from specialTools.threePhasesNNModel import threePhasesModel
from specialTools.threePhasesNNModel import mcc_metric

train = pd.read_csv('./processedData/train_DL.csv')
test = pd.read_csv('./processedData/test_DL.csv')

train_features = np.load('./processedData/train_DL_features.npy')
test_features = np.load('./processedData/test_DL_features.npy')

test_target = np.load('./processedData/test_DL_target.npy')


train['features'] = list(train_features)
test['features'] = list(test_features)
test['target'] = list(test_target)


print(train.features[0].shape)

test.head()

del train_features
del test_features
gc.collect()



seq_len = train.features[0].shape[0]
features_dims = train.features[0].shape[1]

test_target_dim = test.target[0].shape[0]


modellist = [semi_superviesd_LSTM_selfAttention, semi_superviesd_LSTM_selfAttention, semi_superviesd_LSTM_selfAttention]
modelParamList = [
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'unsupervised_out_dim': test_target_dim,
            'learning_rate': 0.0003,
            'unsupervised_lr': 0.0005,

            'lstm_layers': 2,
            'lstm_out_dim': 80,
            'gru_layers': 1,
            'gru_out_dim':  50,

            'selfAttentionSeq_dim': 0,

            'selfAttention_dim': 200,
            'linearReduction_dim': 64,

            'env': 'phase_0',
        },
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'unsupervised_out_dim': test_target_dim,
            'learning_rate': 0.0003,
            'unsupervised_lr': 0.0005,

            'lstm_layers': 2,
            'lstm_out_dim': 80,
            'gru_layers': 1,
            'gru_out_dim':  50,

            'selfAttentionSeq_dim': 0,

            'selfAttention_dim': 200,
            'linearReduction_dim': 64,

            'env': 'phase_1',
        },
        {
            'features_dims':features_dims,
            'seq_len': seq_len,
            'unsupervised_out_dim': test_target_dim,
            'learning_rate': 0.0003,
            'unsupervised_lr': 0.0005,

            'lstm_layers': 2,
            'lstm_out_dim': 80,
            'gru_layers': 1,
            'gru_out_dim':  50,

            'selfAttentionSeq_dim': 0,

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

model = threePhasesModel(modellist, modelParamList, phaseList)
model.fit(train, kfold, train_epochs_list, batch_size_list)
          # test=test, test_epochs=10, test_batch_size=32)

score, threshold = mcc_metric(model.train_total_target, model.train_total_preds)
print('total score:\t', score, 'threshold:\t', threshold)

answer = model.predict_proba(test)

answer.target = (answer.target > threshold).astype(int)
answer.to_csv('./submit.csv', index=False)

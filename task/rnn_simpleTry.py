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

answerlist = []
for i in test.phase.unique().tolist():
    train_x = train[train.phase==i].features
    train_x = np.array(train_x.tolist())
    train_y = train[train.phase==i].target.values

    test_x = test[test.phase==i].features
    test_x = np.array(test_x.tolist())

    test_id = test[test.phase==i].signal_id

    seq_len = train.features[0].shape[0]
    features_dims = train.features[0].shape[1]


    fold_num = 5
    seed_start = 10086

    train_epochs = 25
    batch_size = 32

    test_pred = np.zeros((test_x.shape[0], fold_num))
    train_preds = np.zeros((len(train_x)))
    kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=10)
    for i, (train_idx, valid_idx) in enumerate(kfold.split(train_x, train_y)):

        x_train_fold = train_x[train_idx]
        y_train_fold = train_y[train_idx, np.newaxis]
        x_val_fold = train_x[valid_idx]
        y_val_fold = train_y[valid_idx, np.newaxis]


        model = LSTM_softmax(seed_start+i*500, features_dims, seq_len)
        model = model.cuda()

        model.fit(x_train_fold, y_train_fold, train_epochs, batch_size, x_val_fold, y_val_fold,
                  custom_metric=roc_auc_score)

        test_pred[:, i] = model.predict_proba(test_x)

        train_preds[valid_idx] = model.predict_proba(x_val_fold)
        print('=='*25)

    score, threshold = mcc_metric(train_y, train_preds)
    print('score:\t', score, 'threshold:\t', threshold)

    print('test corr:')
    print(pd.DataFrame(test_pred).corr())

    answer = pd.DataFrame({'signal_id': test_id, 'target':(test_pred.mean(axis=1)>threshold).astype(int)})
    answerlist.append(answer)

answer_out = pd.concat(answerlist)
answer_out.to_csv('./submit.csv', index=False)

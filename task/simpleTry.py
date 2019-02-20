import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.manifold import TSNE as tsne

from lxyTools.adversarialValidation import adversarialValidation

train = pd.read_csv('./processedData/train.csv')
test = pd.read_csv('./processedData/test.csv')

answer = pd.read_csv('./rawdata/sample_submission.csv')


train_x = train.drop(['signal_id', 'id_measurement', 'target'], axis=1)
train_y = train['target']

# train_x =  train_x.drop(train_x.columns[4:7], axis=1)


test_x = test[train_x.columns]
test_id = test.signal_id


re = adversarialValidation(train_x, test_x)

plt.figure(figsize=(12, 5))
plt.bar(re.index, re.values)
plt.show()

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


train_pred = np.zeros(train.shape[0])
test_pred = np.zeros(test.shape[0])

fold_num=5
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=10)
for i, (train_idx, valid_idx) in enumerate(kfold.split(train_x, train_y)):

    x_train_fold = train_x.iloc[train_idx]
    y_train_fold = train_y[train_idx]
    x_val_fold = train_x.iloc[valid_idx]
    y_val_fold = train_y[valid_idx]

    model = lgb.LGBMClassifier(n_estimators=200,
                               learning_rate=0.05,
                               num_leaves=31,
                               min_child_samples=10,
                               max_depth=-1,
                               random_state=2017)

    model.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
              verbose=20, eval_metric='auc')

    val_pred = model.predict_proba(x_val_fold)[:, 1]
    train_pred[valid_idx] = val_pred

    score, threshold = mcc_metric(y_val_fold, val_pred)
    print('score:\t', score, '\tthreshold:\t', threshold)

    test_pred = test_pred + model.predict_proba(test_x)[:, 1]/fold_num

    print('---'*50)


score, threshold = mcc_metric(train_y, train_pred)
print('score:\t', score, '\tthreshold:\t', threshold)

plt.figure(figsize=(15,5))
plt.subplot(121)
sns.distplot(train_pred, bins=20, kde=False)
plt.subplot(122)
sns.distplot(test_pred, bins=20, kde=False)
plt.show()


print(((test_pred > threshold) == 1).sum())

answer.signal_id = test_id
answer.target = (test_pred > threshold).astype(int)
answer.to_csv('./submit.csv', index=False)
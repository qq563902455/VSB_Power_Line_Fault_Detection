import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.manifold import TSNE as tsne

from lxyTools.adversarialValidation import adversarialValidation
from specialTools.kfold import StratifiedKFold
from lxyTools.singleModelUtils import singleModel


train = pd.read_csv('./processedData/train.csv')
test = pd.read_csv('./processedData/test.csv')

answer = pd.read_csv('./rawdata/sample_submission.csv')

train_x = train.drop(['signal_id', 'id_measurement', 'target'], axis=1)

train_y = train['target']

train_x.shape


test_x = test[train_x.columns]
test_id = test.signal_id


adv_feature_importances = adversarialValidation(train_x, test_x,
                                                kfold=StratifiedKFold(
                                                pd.concat([train.id_measurement,
                                                           test.id_measurement]),
                                                n_splits=5,
                                                random_state=0, shuffle=True))

adv_feature_importances.sort_values().tail(50)


plt.figure(figsize=(12, 5))
plt.bar(adv_feature_importances.index, adv_feature_importances.values)
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


fold_num=5
kfold = StratifiedKFold(train.id_measurement, n_splits=fold_num, shuffle=True, random_state=10)
model = lgb.LGBMClassifier(n_estimators=750,
                           # objective='binary',
                           learning_rate=0.02,
                           num_leaves=20,
                           min_child_samples=35,
                           max_depth=6,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           reg_alpha=0.0,
                           reg_lambda=0.0,
                           random_state=2017)
model = singleModel(model, kfold=kfold)
model.fit(train_x, train_y, metric=lambda x,y: mcc_metric(x, y[:, 1])[0],
          train_pred_dim=2, eval_set_param_name='eval_set', eval_metric='auc',
          verbose=100)

feature_importances = pd.Series(model.get_feature_importances())
feature_importances.index = train_x.columns

plt.figure(figsize=(12, 5))
plt.bar(feature_importances.index, feature_importances.values)
plt.show()

feature_processed_importances = feature_importances - 1.0*adv_feature_importances
feature_processed_importances = feature_processed_importances.sort_values()



feature_selected = feature_processed_importances.tail(50).index

train_x_selected = train_x[feature_selected]
test_x_selected = test_x[feature_selected]


print('--'*20,'after feature selection', '--'*20)
adversarialValidation(train_x_selected, test_x_selected,
                      kfold=StratifiedKFold(
                      pd.concat([train.id_measurement,
                                test.id_measurement]),
                      n_splits=5,
                      random_state=0, shuffle=True))


fold_num=5
kfold = StratifiedKFold(train.id_measurement, n_splits=fold_num, shuffle=True, random_state=10)
model = lgb.LGBMClassifier(n_estimators=750,
                           # objective='binary',
                           learning_rate=0.02,
                           num_leaves=20,
                           min_child_samples=35,
                           max_depth=6,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           min_split_gain=0.0000,
                           reg_alpha=0.00,
                           reg_lambda=0.02,
                           random_state=2017)
model = singleModel(model, kfold=kfold)
model.fit(train_x_selected, train_y, metric=lambda x,y: mcc_metric(x, y[:, 1])[0],
          train_pred_dim=2, eval_set_param_name='eval_set', eval_metric='auc',
          verbose=100)

print('--'*20, 'calculating threshold', '--'*20)
train_pred = model.train_pred[:, 1]
score, threshold = mcc_metric(train_y, train_pred)
print('score:\t', score, '\tthreshold:\t', threshold)

print('--'*20, 'predicting test dataset', '--'*20)
test_pred = model.predict_proba(test_x_selected)[:, 1]

print('--'*20, 'ploting the distribution of prediction', '--'*20)
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.distplot(train_pred, bins=20, kde=False)
plt.subplot(122)
sns.distplot(test_pred, bins=20, kde=False)
plt.title('distribution of prediction')
plt.show()


print('number of positive samples:\t', ((test_pred > threshold) == 1).sum())


answer.signal_id = test_id
answer.target = (test_pred > threshold).astype(int)
answer.to_csv('./submit.csv', index=False)

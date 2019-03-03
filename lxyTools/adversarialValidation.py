import pandas as pd
import numpy as np
from tqdm import tqdm


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lxyTools.singleModelUtils import singleModel

def adversarialValidation(train, test,
                          kfold=StratifiedKFold(n_splits=5,
                                random_state=0, shuffle=True),
                          model=-1):
    isTest = np.concatenate([np.zeros(train.shape[0]),
                             np.ones(test.shape[0])], axis=0)

    isTest = pd.Series(isTest)
    totaldata = pd.concat([train, test], axis=0)
    if type(model) == int:
        model=lgb.LGBMClassifier(random_state=2019)

    model = singleModel(model, kfold=kfold)
    model.fit(totaldata, isTest, metric=lambda x,y: roc_auc_score(x, y[:, 1]), train_pred_dim=2)

    importances = model.get_feature_importances()
    importances = pd.Series(importances)
    importances.index = totaldata.columns

    return importances, model.train_pred[:train.shape[0], 1]

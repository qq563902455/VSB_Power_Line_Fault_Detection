# -*- coding: utf-8 -*-
import numpy as np
import copy as cp
from sklearn.model_selection import StratifiedKFold

class singleModel:
    def __init__(self, model, proba=True,
                 kfold=StratifiedKFold(n_splits=5,
                                       random_state=0,
                                       shuffle=True)):
        self.proba = proba
        self.kfold = kfold
        self.model = model

    def fit(self, X, Y, metric, addX=-1, addY=-1):
        self.modelList = []
        self.scorelist = []
        for kTrainIndex, kTestIndex in self.kfold.split(X, Y):
            kTrain_x = X.iloc[kTrainIndex]
            kTrain_y = Y.iloc[kTrainIndex]

            kTest_x = X.iloc[kTestIndex]
            kTest_y = Y.iloc[kTestIndex]

            if type(addX) == type(X):
                kTrain_x = kTrain_x.append(addX)
                kTrain_y = kTrain_y.append(addY)

            model = cp.deepcopy(self.model)
            model.fit(kTrain_x, kTrain_y)

            if self.proba:
                pre = model.predict_proba(kTest_x)
            else:
                pre = model.predict(kTest_x)

            score = metric(kTest_y, pre)
            print('score: ', score)

            self.modelList.append(model)
            self.scorelist.append(score)

        print('mean score: ', np.array(self.scorelist).mean())
        print('std score: ', np.array(self.scorelist).std())
        print('-'*20)

    def predict_proba(self, X):
        return self.predict(X, proba=True)

    def predict(self, X, proba=False):
        out = 0
        for model in self.modelList:
            if isinstance(type(out), type(0)):
                if proba:
                    out = model.predict_proba(X)
                else:
                    out = model.predict(X)
            else:
                if proba:
                    out += model.predict_proba(X)
                else:
                    out += model.predict(X)

        out = out/len(self.modelList)
        return out
        
    def get_feature_importances(self):
        result = np.zeros(self.modelList[0].feature_importances_.shape[0])
        for model in self.modelList:
            result = result + model.feature_importances_
        return result

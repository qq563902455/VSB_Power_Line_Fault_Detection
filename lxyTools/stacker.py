# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:07:42 2017

@author: 56390
"""

import pandas as pd;
import numpy  as np;
import copy as cp;
from sklearn.model_selection import StratifiedKFold;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import cross_val_score;
import gc


class stacker:
    def __init__(self,modelList,higherModel=-1,obj='binary',kFold=StratifiedKFold(n_splits=5,random_state=0),kFoldHigher=StratifiedKFold(n_splits=5,random_state=777)):

        #assert type(modelList)==list,"输入必须为一个模型的列表"
        #assert type(kFold)==StratifiedKFold,"输入必须为StratifiedKFold"

        self.kFold=kFold;
        self.kFoldHigher=kFoldHigher;
        self.modelList=modelList;
        self.higherModel=higherModel;
        self.obj=obj;


    def fit(self,X,Y,metric,featuresList=-1,upsample=False,addX=-1,addY=-1):

        #assert type(X)==pd.DataFrame,"X输入必须为DataFrame"
        #assert type(Y)==pd.DataFrame,"Y输入必须为DataFrame"
        self.modelScoreList=[];
        self.modelListList=[];
        self.modelHigherList=[];


        if featuresList==-1:
            featuresList=[];
            for i in range(len(self.modelList)):
                featuresList.append(X.columns.tolist());

        self.higherTrain=pd.DataFrame(np.zeros((Y.shape[0], len(self.modelList))));
        i=0;
        for model in self.modelList:
            modelList=[];
            ScoreList=[];

            featuresUsed=featuresList[self.modelList.index(model)];

            for kTrainIndex,kTestIndex in self.kFold.split(X,Y):
                kTrain_x=X[featuresUsed].iloc[kTrainIndex];
                kTrain_y=Y.iloc[kTrainIndex];

                if upsample:
                    upsampleTrainData=self.upsample(kTrain_x,kTrain_y);
                    kTrain_x=upsampleTrainData.drop(['label'],axis=1);
                    kTrain_y=upsampleTrainData['label'];


                kTest_x=X[featuresUsed].iloc[kTestIndex];
                kTest_y=Y.iloc[kTestIndex];

                if type(addX) == type(X):
                    kTrain_x = kTrain_x.append(addX[featuresUsed])
                    kTrain_y = kTrain_y.append(addY)

                modelCp=cp.deepcopy(model);
                modelCp.fit(kTrain_x,kTrain_y);
                gc.collect()
                modelList.append(modelCp);

                if self.obj=='binary':
                    testPre=modelCp.predict_proba(kTest_x)[:,1];
                elif self.obj=='reg':
                    testPre=modelCp.predict(kTest_x);
                score=metric(kTest_y, testPre)
                ScoreList.append(score);
                print('score: ',score);
                self.higherTrain.values[kTestIndex,i]=testPre;


            self.modelScoreList.append((np.array(ScoreList).mean(),np.array(ScoreList).std()));
            print('mean score: ',np.array(ScoreList).mean());
            print('std       : ',np.array(ScoreList).std())
            print('-'*20);
            i+=1;
            self.modelListList.append(modelList);


        print(self.higherTrain.shape);

        self.re_score=[];

        for kTrainIndex,kTestIndex in self.kFoldHigher.split(self.higherTrain,Y):
            higherModelcp=cp.deepcopy(self.higherModel);
            kTrain_x=self.higherTrain.iloc[kTrainIndex];
            kTrain_y=Y.iloc[kTrainIndex];

            kTest_x=self.higherTrain.iloc[kTestIndex];
            kTest_y=Y.iloc[kTestIndex];
            higherModelcp.fit(kTrain_x,kTrain_y);
            gc.collect()

            if self.obj=='binary':
                testPre=higherModelcp.predict_proba(kTest_x)[:,1];
            elif self.obj=='reg':
                testPre=higherModelcp.predict(kTest_x);

            score=metric(kTest_y, testPre)
            print('score: ',score)
            self.re_score.append(score);
            self.modelHigherList.append(higherModelcp);

        print('stacker score',np.array(self.re_score).mean());
        print('stacker std  ',np.array(self.re_score).std());


    def highModelRefit(self, Y, metric):
        self.modelHigherList=[];
        for kTrainIndex,kTestIndex in self.kFoldHigher.split(self.higherTrain,Y):
            higherModelcp=cp.deepcopy(self.higherModel);
            kTrain_x=self.higherTrain.iloc[kTrainIndex];
            kTrain_y=Y.iloc[kTrainIndex];

            kTest_x=self.higherTrain.iloc[kTestIndex];
            kTest_y=Y.iloc[kTestIndex];
            higherModelcp.fit(kTrain_x,kTrain_y);
            gc.collect()

            if self.obj=='binary':
                testPre=higherModelcp.predict_proba(kTest_x)[:,1];
            elif self.obj=='reg':
                testPre=higherModelcp.predict(kTest_x);

            score=metric(kTest_y, testPre)
            print('score: ',score)
            self.re_score.append(score);
            self.modelHigherList.append(higherModelcp);

        print('stacker score',np.array(self.re_score).mean());
        print('stacker std  ',np.array(self.re_score).std());
        pass

    def upsample(self,X,Y):
#        print('X shape: ',X.shape);
#        print('Y shape: ',Y.shape);

        data_copy=X.copy(deep=True);
        data_copy['label']=Y;

        positiveData=data_copy[data_copy['label']==1];
        negativeData=data_copy[data_copy['label']==0];

        pNum=positiveData.shape[0];
        nNum=negativeData.shape[0];

#        print('原始负样本数量',nNum);
#        print('原始正样本数量',pNum);


        if pNum>nNum:
            ratio=(int)(pNum/nNum);
            for i in range(0,ratio-1):
                data_copy=data_copy.append(negativeData,ignore_index=True);
        elif nNum>pNum:
            ratio=(int)(nNum/pNum);
            for i in range(0,ratio-1):
                data_copy=data_copy.append(positiveData,ignore_index=True);
#        print(data_copy.shape);
#        print('负样本数量',data_copy[data_copy['label']==0].shape[0]);
#        print('正样本数量',data_copy[data_copy['label']==1].shape[0]);

        return data_copy;

    def predict(self,X,featuresList=-1):
        #assert type(X)==pd.DataFrame,"X输入必须为DataFrame"

        if featuresList==-1:
            featuresList=[];
            for i in range(len(self.modelList)):
                featuresList.append(X.columns.tolist());
        ans=0;
        higherX=pd.DataFrame();
        for i in range(0,len(self.modelListList)):
            featuresUsed=featuresList[i];
            for cf in self.modelListList[i]:
                if i not in higherX.columns:
                    higherX[i]=cf.predict(X[featuresUsed])/len(self.modelListList[i]);
                else:
                    higherX[i]+=cf.predict(X[featuresUsed])/len(self.modelListList[i]);
        ans=0;
        for higherModel in self.modelHigherList:
            if type(ans)==type(0):
                ans=higherModel.predict(higherX)/len(self.modelHigherList);
            else:
                ans+=higherModel.predict(higherX)/len(self.modelHigherList);
        return ans;


    def predict_proba(self,X,featuresList=-1):
        #assert type(X)==pd.DataFrame,"X输入必须为DataFrame"
        if featuresList==-1:
            featuresList=[];
            for i in range(len(self.modelList)):
                featuresList.append(X.columns.tolist());
        ans=0;
        higherX=pd.DataFrame();
        for i in range(0,len(self.modelListList)):
            featuresUsed=featuresList[i];
            for cf in self.modelListList[i]:
                if i not in higherX.columns:
                    higherX[i]=cf.predict_proba(X[featuresUsed])[:,1]/len(self.modelListList[i]);
                else:
                    higherX[i]+=cf.predict_proba(X[featuresUsed])[:,1]/len(self.modelListList[i]);
        ans=0;
        for higherModel in self.modelHigherList:
            if type(ans)==type(0):
                ans=higherModel.predict_proba(higherX)[:,1]/len(self.modelHigherList);
            else:
                ans+=higherModel.predict_proba(higherX)[:,1]/len(self.modelHigherList);
        return ans;

class linearBlending:
    def __init__(self,paramList,sum_counts,loss,obj):
        self.paramList=paramList;
        self.sum_counts=sum_counts;
        self.bestScore=-1;
        self.loss=loss;
        self.obj=obj;
        self.bestParam=cp.deepcopy(paramList);
    def __enumerate(self,sum_counts,paramNum,X,Y):
        paramNum-=1;
        if paramNum>0:
            temp=sum_counts;
            for i in range(0,sum_counts+1):
                self.paramList[paramNum]=i;
                temp=sum_counts-i;
                self.__enumerate(temp,paramNum,X,Y);
        else:
            self.paramList[0]=sum_counts;
            if self.obj=='classification':
                pre=self.predict_proba(X)[:,1];
            elif self.obj=='reg':
                pre=self.predict(X);
            score=self.loss(Y,pre);

            if score>self.bestScore:
                self.bestScore=score;
                self.bestParam=cp.deepcopy(self.paramList);
    def fit(self,X,Y):
        self.bestScore=-1;
        self.__enumerate(self.sum_counts,len(self.paramList),X,Y);
        self.paramList=cp.deepcopy(self.bestParam);
    def predict_proba(self,X):
        ans=0;
        X=pd.DataFrame(X);
        for i in range(0,len(self.paramList)):
            if type(ans)==type(0):
                ans=X.values[:,i]*self.paramList[i]/self.sum_counts;
            else:
                ans+=X.values[:,i]*self.paramList[i]/self.sum_counts;
        #print(np.vstack((1-ans,ans)).shape);
        return np.vstack((1-ans,ans)).T;

    def predict(self,X):
        if self.obj=='reg':
            ans=0;
            X=pd.DataFrame(X);
            for i in range(0,len(self.paramList)):
                if type(ans)==type(0):
                    ans=X.values[:,i]*self.paramList[i]/self.sum_counts;
                else:
                    ans+=X.values[:,i]*self.paramList[i]/self.sum_counts;
            return ans;

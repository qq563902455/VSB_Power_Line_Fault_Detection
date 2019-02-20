# -*- coding: utf-8 -*-
import numpy as np;
import pandas as pd;
from minepy import MINE;  
import copy as cp;

def maximalInformationCoef(x,y):
    scoreList=[];    
    xc=pd.DataFrame(x);
    yc=pd.Series(y);
    
    print('x shape:\t',xc.shape);
    for col in xc.columns:
        print(col)
        m = MINE();
        m.compute_score(xc[col],yc)
        scoreList.append(m.mic());
        
    score=np.array(scoreList);
    print(score);
    print('max mic:\t',score.max());
    print('min mic:\t',score.min());
    print('mean mic:\t',score.mean());    
    return score;
def relationCoef(x,y,fun):
    scoreList=[];    
    xc=pd.DataFrame(x);
    yc=pd.Series(y);
    
    print('x shape:\t',xc.shape);
    for col in xc.columns:
        scoreList.append(fun(xc[col],yc));
        
    score=np.array(scoreList);
    print(score);
    print('max:\t',score.max());
    print('min:\t',score.min());
    print('mean:\t',score.mean());    
    return score;

def micCompute(x,y):
    m = MINE();
    m.compute_score(x,y);
    return m.mic();


def rfeByMultiModel(modellist,step,objnum,x,y,selectway=-1):
    modelcplist=[];
    for model in modellist:
        modelcp=cp.deepcopy(model);
        modelcplist.append(modelcp);

    x_remain=x.copy();
    while True:
        print('fitting with ',x_remain.shape[1],' features');
        featureSelectedlistlist=[];
        featureSelected=x_remain.columns.tolist();
        for model in modelcplist:
            model.fit(x_remain,y);
            importancelist=pd.Series(model.feature_importances_);
            importancelist.index=x_remain.columns;
            if x_remain.shape[1]-step>objnum:
                selectedlist=importancelist.sort_values().tail(x_remain.shape[1]-step).index.tolist();
            else:
                selectedlist=importancelist.sort_values().tail(objnum).index.tolist();
            featureSelectedlistlist.append(selectedlist);

        if selectway==-1:
            for featurelist in featureSelectedlistlist:
                for fe in featureSelected.copy():
                    if fe not in featurelist:
                        featureSelected.remove(fe);
        else:
            featureSelected=selectway(featureSelectedlistlist);


        if len(featureSelected)<objnum:
            return  featureSelected;
        x_remain=x_remain[featureSelected];
        
        
def rfeBySingleModel(model, step, objnum, X, y, valid, metric):
    x_remain = X.copy()
    validX = valid[0].copy()
    scorelist = []
    bestScore = 100
    bestFeatures = []
    while True:
       print('fitting with ',x_remain.shape[1],' features')
       model.fit(x_remain, y, metric)
       score = metric(valid[1], model.predict_proba(validX)[:, 1])
       print('score:\t', score)
       if score < bestScore:
           bestScore = score
           bestFeatures = x_remain.columns
       scorelist.append(score)
       if x_remain.shape[1] <= objnum:
           break
       
       featureImportancelist = []
       for m in model.modelList:
            featureImportancelist.append(m.feature_importances_)
       importancelist = pd.Series(featureImportancelist[0])
       for i in range(1, len(featureImportancelist)):
            importancelist += pd.Series(featureImportancelist[i])
       importancelist.index = x_remain.columns
        
       droplist = importancelist.sort_values().head(step).index.tolist()
       print('drop:\t', droplist)
       x_remain = x_remain.drop(droplist, axis=1)
       validX = validX.drop(droplist, axis=1)
    return (scorelist, bestFeatures)    
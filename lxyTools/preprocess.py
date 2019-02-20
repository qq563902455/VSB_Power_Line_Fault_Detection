# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:04:08 2017

@author: 56390
"""
import pandas as pd;
import numpy  as np;
from sklearn.preprocessing import OneHotEncoder;
from sklearn.preprocessing import MinMaxScaler;

class rawDataProcess:
    def __init__(self,train_data,test_train):

        assert type(train_data)==pd.DataFrame,'训练数据类型必须为DataFrame';
        assert type(test_train)==pd.DataFrame,'测试数据类型必须为DataFrame';

        self.train_data=train_data;
        self.test_data=test_train;
        self.dataset=self.train_data.copy().append(self.test_data.copy()).reset_index(drop=True);
    def dropfeatures(self,dropfun=-1):
        if dropfun==-1:
            '''
            删除全为空的列
            '''
            temp=self.dataset.count();
            nonelist=temp[temp==0].index.tolist();
            print('num of nonefeatures: ',len(nonelist));
            self.dataset=self.dataset.drop(nonelist,axis=1);
            '''
            删除所有值都为同一个值的列
            '''
            zeroStdList=[];
            for col in self.dataset.columns:
                if self.dataset[col].max()==self.dataset[col].min():
                    zeroStdList.append(col);
            print('num of noneVaryFeatures:　',len(zeroStdList));
            self.dataset=self.dataset.drop(zeroStdList,axis=1);
        else:
            self.dataset=dropfun(self.dataset);

    def fillna(self,fillfun=-1):

        if fillfun==-1:
            temp=self.dataset.count();
            nalist=temp[temp<self.dataset.shape[0]];
            nalist=nalist[nalist>0].index.tolist();
            print('数据列数:　',self.dataset.shape[1]);
            print('含有空值列数(非全空):　',len(nalist));

            for col in nalist:
                self.dataset[col+'_na']=0;
                self.dataset.loc[self.dataset[col].isnull(),col+'_na']=1;
                if self.dataset[col].dtype!='object':
                    self.dataset.loc[self.dataset[col].isnull(),col]=self.dataset[col].mean();
                else:
                    self.dataset.loc[self.dataset[col].isnull(),col]=self.dataset[col].mode()[0];
        else:
            self.dataset=fillfun(self.dataset);
    def cat_sc_fit(self,colCatList,colScList):

        assert type(colCatList)==list,'CAT类型数据所在列列表必须为list';
        assert type(colScList)==list, '需要尺度调整的数据所在列列表必须为list';

        self.enc=OneHotEncoder(dtype=np.int32);
        self.minMax=MinMaxScaler();
        self.colCatList=colCatList;
        self.colScList=colScList;

        self.colCatNameList=[];
        temp=self.dataset[colCatList].copy();

        for col in colCatList:
            i=0;
            for val in  self.dataset[col].unique():
                i=i+1;
                temp.loc[self.dataset[col]==val,col]=i;
                self.colCatNameList.append(col+'_'+str(val));
        self.dataset[colCatList]=temp.values;
        print('len colCatNameList:　',len(self.colCatNameList));

        if len(colCatList)>=1:
            self.enc.fit(temp);
        if len(colScList)>=1:
            self.minMax.fit(self.dataset[colScList]);
    def toTrainData(self):

        allList=self.colCatList+self.colScList;
        train_data=self.dataset.iloc[0:self.train_data.shape[0]];
        print('train_data size: ',train_data.shape);
        dataRe=train_data.drop(allList,axis=1);
        if len(self.colCatList)>=1:
            encTransData=pd.DataFrame(self.enc.transform(train_data[self.colCatList]).toarray());
            dataRe[self.colCatNameList]=encTransData;

        if len(self.colScList)>=1:
            scTransData=pd.DataFrame(self.minMax.transform(train_data[self.colScList]));
            dataRe[self.colScList]=scTransData;
        print('re shape: ',dataRe.shape);
        return dataRe;

    def toTestData(self):
        allList=self.colCatList+self.colScList;
        test_data=self.dataset.iloc[self.train_data.shape[0]:];
        print('test_data size: ',test_data.shape);
        dataRe=test_data.drop(allList,axis=1);
        if len(self.colCatList)>=1:
            encTransData=pd.DataFrame(self.enc.transform(test_data[self.colCatList]).toarray());
            dataRe[self.colCatNameList]=encTransData;

        if len(self.colScList)>=1:
            scTransData=pd.DataFrame(self.minMax.transform(test_data[self.colScList]));
            dataRe[self.colScList]=scTransData;
        print('re shape: ',dataRe.shape);
        return dataRe;

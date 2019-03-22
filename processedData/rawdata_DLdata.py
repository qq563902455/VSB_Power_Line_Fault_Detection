import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyarrow.parquet as pq
import os
import random
import math
import gc
import pywt

from scipy.signal import welch
from scipy.signal import find_peaks

from lxyTools.process import mutiProcessLoop

def removeRPI(peaks, x):
    peaks = sorted(peaks)
    peaks_num = len(peaks)
    peaks_pd = []

    for i in range(peaks_num):
        peak = peaks[i]
        peak_height = x[peaks[i]]

        if i!=0:
            left_peak = peaks[i-1]
            left_peak_height = x[peaks[i-1]]
            if  peak - left_peak > 10000:
                continue

        if i!=peaks_num-1:
            right_peak = peaks[i+1]
            right_peak_height = x[peaks[i+1]]
            if  right_peak - peak > 10000:
                continue

        peaks_pd.append(peak)

    return peaks_pd



def peakStatistics(data, std_ratio=5):
    re = np.zeros((data.shape[1], 9))
    for col in range(data.shape[1]):
        x = data[:, col]
        height = x.std()*std_ratio


        peaks, peaks_info = find_peaks(x, height=height, distance=1000)
        peaks = removeRPI(peaks, x)

        neg_peaks, neg_peaks_info = find_peaks(-x, height=height, distance=1000)
        neg_peaks = removeRPI(neg_peaks, x)

        peaks_height = x[peaks+neg_peaks]

        if peaks_height.shape[0] > 0:
            re[col, 0] = peaks_height.min()
            re[col, 1] = peaks_height.max()
            re[col, 2] = peaks_height.shape[0]/1000

            if peaks_height.shape[0] > 1:
                re[col, 4] = np.percentile(peaks_height, 1)
                re[col, 5] = np.percentile(peaks_height, 99)
                re[col, 6] = np.percentile(peaks_height, 25)
                re[col, 7] = np.percentile(peaks_height, 75)
                re[col, 8] = np.percentile(peaks_height, 50)
    return re

def extractTestTarget(rawdata):
    out_df = pd.DataFrame()
    out_df['signal_id'] = rawdata.columns.astype(int)

    rollingdata = rawdata.rolling(20000, center=True, min_periods=1, axis=0)
    trend = rollingdata.mean()
    res = (rawdata - trend).values
    # res = np.diff(rawdata.values, n=1, axis=0)
    res = (2*(res+128)/255)-1

    # peaks_info_5std = peakStatistics(res, 5)
    peaks_info_10std = peakStatistics(res, 10)

    out = np.concatenate([peaks_info_10std], axis=1)

    out_df['target'] = list(out)
    return out_df




def extractFeatures(rawdata):
    out_df = pd.DataFrame()
    out_df['signal_id'] = rawdata.columns.astype(int)


    rollingdata = rawdata.rolling(20000, center=True, min_periods=1, axis=0)
    trend = rollingdata.mean()
    res = (rawdata - trend).values


    res = (2*(res+128)/255)-1

    whole_signal_len = 800000
    processed_len = 250
    bin_len = int(whole_signal_len/processed_len)

    slice_list = []
    for i in range(0, whole_signal_len, bin_len):
        signal_slice = res[i:(i+bin_len), :]


        mean_slice = signal_slice.mean(axis=0)
        std_slice = signal_slice.std(axis=0)

        std_top = mean_slice + std_slice
        std_bot = mean_slice - std_slice

        percentil_slice = np.percentile(signal_slice, [0, 1, 50, 99, 100], axis=0)
        max_range = percentil_slice[-1] - percentil_slice[0]

        relative_percentile = percentil_slice - mean_slice


        out_slice = np.concatenate([
                            percentil_slice.T,
                            max_range.reshape(-1, 1),
                            relative_percentile.T,
                            std_top.reshape(-1, 1),
                            std_bot.reshape(-1, 1),
                            std_slice.reshape(-1, 1),
                            mean_slice.reshape(-1, 1)
                            ], axis=1)
        slice_list.append(out_slice)

    out = np.array(slice_list).transpose([1, 0, 2])
    out_df['features'] = list(out)
    return out_df

def readRawSignal_extractFeatures(path, subset_size=50, start_id=0, end_id=29049):
    relist = []

    processFun = lambda x: extractFeatures(
                pq.read_pandas(
                    path,
                    columns=[str(val) for val in range(start_id+x*subset_size, min(start_id+(x+1)*subset_size, end_id))]).to_pandas()
                )
    multiProcess = mutiProcessLoop(processFun, range(math.ceil((end_id-start_id)/subset_size)), n_process=4, silence=False)
    resultlist = multiProcess.run()
    return pd.concat(resultlist)

train_features = readRawSignal_extractFeatures('./rawdata/train.parquet',subset_size=100, start_id=0, end_id=8712)
test_features = readRawSignal_extractFeatures('./rawdata/test.parquet',subset_size=100, start_id=8712, end_id=29049)








def readRawSignal_extractTestTarget(path, subset_size=50, start_id=8712, end_id=29049):
    relist = []

    processFun = lambda x: extractTestTarget(
                pq.read_pandas(
                    path,
                    columns=[str(val) for val in range(start_id+x*subset_size, min(start_id+(x+1)*subset_size, end_id))]).to_pandas()
                )
    multiProcess = mutiProcessLoop(processFun, range(math.ceil((end_id-start_id)/subset_size)), n_process=4, silence=False)
    resultlist = multiProcess.run()
    return pd.concat(resultlist)


test_targets = readRawSignal_extractTestTarget('./rawdata/test.parquet',subset_size=100, start_id=8712, end_id=29049)


gc.collect()

train_meta = pd.read_csv('./rawdata/metadata_train.csv')
test_meta = pd.read_csv('./rawdata/metadata_test.csv')

def threePhasesConcat(features, meta):
    temp = pd.merge(left=features, right=meta[['signal_id', 'id_measurement', 'phase']], on='signal_id', how='left')
    temp =  temp.drop(['signal_id'], axis=1)

    temp = temp.set_index(['id_measurement', 'phase']).unstack('phase')
    tempNp = temp.values

    namelist=[]
    for name in temp.columns:
        namelist.append((str(name[0])+'_'+str(name[1])))

    measure_features = pd.DataFrame(tempNp)
    measure_features.columns = namelist

    measure_features['id_measurement'] = temp.index

    return pd.merge(left=meta, right=measure_features, how='left', on='id_measurement')

train_data = threePhasesConcat(train_features, train_meta)
test_data = threePhasesConcat(test_features, test_meta)

test_data_target = threePhasesConcat(test_targets, test_meta)


targets_values_test = np.array([test_data_target.target_0.tolist(),
                                test_data_target.target_1.tolist(),
                                test_data_target.target_2.tolist()])

targets_values_test = targets_values_test.transpose([1, 0, 2])
targets_values_test =  targets_values_test.reshape(targets_values_test.shape[0], -1)
np.save('./processedData/test_DL_target.npy', targets_values_test)

features_values_train = np.array([train_data.features_0.tolist(),
                                  train_data.features_1.tolist(),
                                  train_data.features_2.tolist()])
features_values_train = features_values_train.transpose([1, 2, 0, 3])


features_values_test = np.array([test_data.features_0.tolist(),
                                 test_data.features_1.tolist(),
                                 test_data.features_2.tolist()])
features_values_test = features_values_test.transpose([1, 2, 0, 3])


features_values_train =  features_values_train.reshape(features_values_train.shape[0], features_values_train.shape[1], -1)
np.save('./processedData/train_DL_features.npy', features_values_train)
train_data.drop(['features_0',
                 'features_1',
                 'features_2'], axis=1).to_csv('./processedData/train_DL.csv', index=False)


features_values_test =  features_values_test.reshape(features_values_test.shape[0], features_values_test.shape[1], -1)
np.save('./processedData/test_DL_features.npy', features_values_test)
test_data.drop(['features_0',
                'features_1',
                'features_2'], axis=1).to_csv('./processedData/test_DL.csv', index=False)

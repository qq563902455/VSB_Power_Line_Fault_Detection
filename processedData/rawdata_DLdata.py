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


#
# def peakStatistics(data, std_ratio=5):
#     re = np.zeros((data.shape[1], 1))
#     for col in range(data.shape[1]):
#         x = data[:, col]
#         height = x.std()*std_ratio
#         peaks, peaks_info = find_peaks(x, height=height, distance=100)
#         neg_peaks, neg_peaks_info = find_peaks(-x, height=height, distance=100)
#         peaks_height = np.concatenate([peaks_info['peak_heights'], neg_peaks_info['peak_heights']])
#
#         if peaks_height.shape[0] > 0:
#             # re[col, 0] = peaks_height.min()
#             # re[col, 1] = peaks_height.max()
#             re[col, 0] = peaks_height.shape[0]/10
#     return re

def extractFeatures(rawdata):
    out_df = pd.DataFrame()
    out_df['signal_id'] = rawdata.columns.astype(int)


    # rawdata_val = (2*(rawdata.values+128)/255)-1
    # diff_1_rawdata = np.diff(rawdata_val, n=1, axis=0)


    rollingdata = rawdata.rolling(20000, center=True, min_periods=1, axis=0)
    trend = rollingdata.mean()
    res = (rawdata - trend).values

    # res = rawdata.values
    # wavelet = 'db4'
    # coeflist = pywt.wavedec(res, wavelet, level=6, axis=0)
    # for i in range(1, len(coeflist)):
    #     coeflist[i] = pywt.threshold(coeflist[i], 5, 'hard')
    # res = pywt.waverec(coeflist, wavelet, axis=0)

    res = (2*(res+128)/255)-1

    whole_signal_len = 800000
    processed_len = 160
    bin_len = int(whole_signal_len/processed_len)

    slice_list = []
    for i in range(0, whole_signal_len, bin_len):
        signal_slice = res[i:(i+bin_len), :]

        # peaks_info_5std = peakStatistics(signal_slice, std_ratio=5)
        # peaks_info_10std = peakStatistics(signal_slice, std_ratio=10)

        mean_slice = signal_slice.mean(axis=0)
        std_slice = signal_slice.std(axis=0)

        std_top = mean_slice + std_slice
        std_bot = mean_slice - std_slice

        percentil_slice = np.percentile(signal_slice, [0, 1, 50, 99, 100], axis=0)
        max_range = percentil_slice[-1] - percentil_slice[0]

        relative_percentile = percentil_slice - mean_slice


        # diff1_slice = diff_1_rawdata[i:(i+bin_len), :]
        # diff1_percentil_slice = np.percentile(diff1_slice, [0, 1, 25, 50, 75, 99, 100], axis=0)

        out_slice = np.concatenate([
                            # peaks_info_5std,
                            # peaks_info_10std,
                            # diff1_percentil_slice.T,
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

features_values_train = np.array([train_data.features_0.tolist(),
                                  train_data.features_1.tolist(),
                                  train_data.features_2.tolist()])
features_values_train = features_values_train.transpose([1, 2, 0, 3])


features_values_test = np.array([test_data.features_0.tolist(),
                                 test_data.features_1.tolist(),
                                 test_data.features_2.tolist()])
features_values_test = features_values_test.transpose([1, 2, 0, 3])


# features_all = np.concatenate([features_values_train, features_values_test], axis=0)
#
# maxval = features_all.max(axis=(0, 1, 2))
# minval = features_all.min(axis=(0, 1, 2))
#
# features_values_train = 2*(features_values_train - minval)/(maxval - minval) - 1
# features_values_test = 2*(features_values_test - minval)/(maxval - minval) - 1

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

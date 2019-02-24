import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import math

from scipy.signal import spectrogram
from scipy.signal import welch
from lxyTools.process import mutiProcessLoop


# 从原始的时间序列中提取特征
# rawdata: 原始的时间序列
# window_size: 在原始序列上面求平均的窗口的大小
# fft_window_size: fft之后的结果上求平均的窗口大小
def signal2features(rawdata, window_size, fft_window_size,
                    stft_window_size, stft_overlap,
                    welch_count_length, welch_count_overlap):
    num_mean_features = int(rawdata.shape[0]/window_size)

    num_fft_mean_features = int(rawdata.shape[0]/fft_window_size/2)

    num_stft_features = int(stft_window_size/2) + 1

    num_add_features = 18

    result = np.zeros((rawdata.shape[1],
                       num_mean_features+num_fft_mean_features+num_stft_features+num_add_features))

    for i in range(rawdata.shape[1]):
        result[i, 0] = int(rawdata.columns[i])
        result[i, 1] = rawdata[rawdata.columns[i]].values.mean()
        result[i, 2] = rawdata[rawdata.columns[i]].values.max()
        result[i, 3] = rawdata[rawdata.columns[i]].values.min()

        diff_data = np.diff(rawdata[rawdata.columns[i]].values, n=1)
        result[i, 4] = diff_data.mean()
        result[i, 5] = diff_data.std()
        result[i, 6] = diff_data.max()
        result[i, 7] = diff_data.min()

        diff_data = np.diff(rawdata[rawdata.columns[i]].values, n=2)
        result[i, 8] = diff_data.mean()
        result[i, 9] = diff_data.std()
        result[i, 10] = diff_data.max()
        result[i, 11] = diff_data.min()

        f, t, Sxx = spectrogram(rawdata[rawdata.columns[i]].values,
                                nperseg=stft_window_size,
                                noverlap=stft_overlap
                                )

        result[i, 12] = Sxx.mean()
        result[i, 13] = Sxx.std()
        result[i, 14] = Sxx.max()

        result[i, num_add_features:(num_add_features+num_stft_features)] = Sxx.mean(axis=1)

        f, Pxx = welch(rawdata[rawdata.columns[i]].values,
                               nperseg=welch_count_length,
                               noverlap=welch_count_overlap)

        result[i, 15] = (Pxx>2.5).sum()
        result[i, 16] = (Pxx>80).sum()
        result[i, 17] = f[np.where(Pxx == Pxx.max())]


        for j in range(num_mean_features):
            result[i, j+num_add_features+num_stft_features] = rawdata[rawdata.columns[i]].values[(j*window_size) : ((j+1)*window_size)].mean()

        fft_re = np.fft.fft(rawdata[rawdata.columns[i]])[:int(rawdata.shape[0]/2)]
        fft_re_abs = np.sqrt(fft_re.real ** 2 + fft_re.imag ** 2)
        for j in range(num_fft_mean_features):
            result[i,  j + num_mean_features + num_add_features + num_stft_features] = fft_re_abs[(j*fft_window_size) : ((j+1)*fft_window_size)].mean()

    result = pd.DataFrame(result)
    result = result.rename({0: 'signal_id',
                            1: 'mean',
                            2: 'max',
                            3: 'min',
                            4: 'diff_1_mean',
                            5: 'diff_1_std',
                            6: 'diff_1_max',
                            7: 'diff_1_min',
                            8: 'diff_2_mean',
                            9: 'diff_2_std',
                            10: 'diff_2_max',
                            11: 'diff_2_min',
                            12: 'stft_mean',
                            13: 'stft_std',
                            14: 'stft_max',
                            15: 'welch>2.5',
                            16: 'welch>80',
                            17: 'max_welch_f',}, axis=1)

    result.signal_id = result.signal_id.astype(int)

    return result

def readRawSignal_extractFeatures(path, subset_size=500, start_id=0, end_id=29049):
    processFun = lambda x: signal2features(
                pq.read_pandas(
                    path,
                    columns=[str(val) for val in range(start_id+x*subset_size, min(start_id+(x+1)*subset_size, end_id))]).to_pandas(),
                160000, 80000,
                stft_window_size=32,
                stft_overlap=8,
                welch_count_length=1024,
                welch_count_overlap=256
                )
    multiProcess = mutiProcessLoop(processFun, range(math.ceil((end_id-start_id)/subset_size)), n_process=4, silence=False)
    resultlist = multiProcess.run()
    return pd.concat(resultlist)

train_features = readRawSignal_extractFeatures('./rawdata/train.parquet',subset_size=2178, start_id=0, end_id=8712)
test_features = readRawSignal_extractFeatures('./rawdata/test.parquet',subset_size=2000, start_id=8712, end_id=29049)

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

# train_data = pd.merge(left=train_features, right=train_meta, how='left', on='signal_id')
# test_data = pd.merge(left=test_features, right=test_meta, how='left', on='signal_id')



train_data.to_csv('./processedData/train.csv', index=False)
test_data.to_csv('./processedData/test.csv', index=False)

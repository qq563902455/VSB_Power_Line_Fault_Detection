import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import math

from scipy import signal
from scipy.signal import spectrogram
from scipy.signal import welch
from lxyTools.process import mutiProcessLoop


# 从原始的时间序列中提取特征
# rawdata: 原始的时间序列
def extractFeatures(rawdata):

    rollingdata = rawdata.rolling(20000, center=True, min_periods=1, axis=0)
    trend = rollingdata.mean()
    res = rawdata - trend

    rolling_res_std = res.rolling(20000, center=True, min_periods=1, axis=0).std()
    rolling_res_std_std = rolling_res_std.std(axis=0)
    rolling_res_std_mean = rolling_res_std.mean(axis=0)


    res_std = res.std(axis=0)
    res_mean = res.mean(axis=0)
    res_abs = (res - res_mean).abs()

    outdata = pd.DataFrame()
    outdata['signal_id'] = rawdata.columns.values.astype(int)

    # outdata['power_rawdata'] = (rawdata.values**2).sum(axis=0)

    # outdata['max_val_trend'] = trend.max(axis=0).values
    # outdata['min_val_trend'] = trend.min(axis=0).values
    outdata['mean_val_trend'] = trend.mean(axis=0).values
    # outdata['power_trend'] = (trend.values**2).sum(axis=0)

    outdata['max_val_res'] = res.max(axis=0).values
    outdata['min_val_res'] = res.min(axis=0).values
    outdata['mean_val_res'] = res_mean.values

    # outdata['power_res'] = (res.values**2).sum(axis=0)

    outdata['50%_val_res'] = np.percentile(res.values, 50, axis=0)

    outdata['rolling_std_res_std'] = rolling_res_std_std.values
    outdata['rolling_std_res_mean'] = rolling_res_std_mean.values
    outdata['rolling_std_res_max'] = rolling_res_std.max(axis=0).values
    outdata['rolling_std_res_min'] = rolling_res_std.min(axis=0).values

    outdata['rolling_std_res>2.5std'] = ((rolling_res_std-rolling_res_std_mean)>(2.5*rolling_res_std_std)).sum(axis=0).values
    outdata['rolling_std_res>5std'] = ((rolling_res_std-rolling_res_std_mean)>(5*rolling_res_std_std)).sum(axis=0).values
    outdata['rolling_std_res>10std'] = ((rolling_res_std-rolling_res_std_mean)>(10*rolling_res_std_std)).sum(axis=0).values

    outdata['val>2.5std_res'] = (res_abs>(2.5*res_std)).sum(axis=0).values
    outdata['val>5std_res'] = (res_abs>(5*res_std)).sum(axis=0).values
    outdata['val>10std_res'] = (res_abs>(10*res_std)).sum(axis=0).values
    outdata['val>20std_res'] = (res_abs>(20*res_std)).sum(axis=0).values

    peaks_count_f = lambda x: len(signal.argrelmax(x, order=1)[0])
    outdata['peak1_num_res'] = np.apply_along_axis(peaks_count_f, axis=0, arr=res)
    peaks_count_f = lambda x: len(signal.argrelmax(x, order=2)[0])
    outdata['peak2_num_res'] = np.apply_along_axis(peaks_count_f, axis=0, arr=res)
    peaks_count_f = lambda x: len(signal.argrelmax(x, order=3)[0])
    outdata['peak3_num_res'] = np.apply_along_axis(peaks_count_f, axis=0, arr=res)

    diff_1_rawdata = np.diff(rawdata.values, n=1, axis=0)
    diff_2_rawdata = np.diff(rawdata.values, n=2, axis=0)

    diff_1_mean = diff_1_rawdata.mean(axis=0)
    diff_1_std = diff_1_rawdata.std(axis=0)
    diff_1_abs = np.abs(diff_1_rawdata-diff_1_mean)

    diff_2_mean = diff_2_rawdata.mean(axis=0)
    diff_2_std = diff_2_rawdata.std(axis=0)
    diff_2_abs = np.abs(diff_2_rawdata-diff_2_mean)


    outdata['max_val_diff1'] = diff_1_rawdata.max(axis=0)
    outdata['min_val_diff1'] = diff_1_rawdata.min(axis=0)
    outdata['std_val_diff1'] = diff_1_std
    outdata['mean_val_diff1'] = diff_1_mean

    outdata['max_val_diff2'] = diff_2_rawdata.max(axis=0)
    outdata['min_val_diff2'] = diff_2_rawdata.min(axis=0)
    outdata['std_val_diff2'] = diff_2_std
    outdata['mean_val_diff2'] = diff_2_mean


    f, Pxx = welch(rawdata.values, nperseg=1024, noverlap=256, axis=0)
    outdata['welch_rawdata>2.5'] = (Pxx>2.5).sum(axis=0)
    outdata['welch_rawdata>80'] = (Pxx>80).sum(axis=0)

    search_f = lambda arr: f[np.where(arr == arr.max())]
    outdata['max_welch_rawdata_f'] = np.apply_along_axis(search_f, axis=0, arr=Pxx).reshape(-1)

    f, t, Sxx = spectrogram(rawdata.values, nperseg=32, noverlap=8, axis=0)
    outdata['max_val_stft'] = Sxx.max(axis=(0,2))
    outdata['mean_val_stft'] = Sxx.mean(axis=(0,2))
    outdata['std_val_stft'] = Sxx.std(axis=(0,2))

    Sxx_mean = Sxx.mean(axis=2)
    for i in range(Sxx_mean.shape[0]):
        outdata['val_'+str(i)+'_stft'] = Sxx_mean[i, :]

    fft_re = np.fft.fft(rawdata.values, axis=0)[0: int(rawdata.shape[0]/2), :]
    fft_re_abs = np.sqrt(fft_re.real ** 2 + fft_re.imag ** 2)

    # outdata['real_7th_harmonic'] = fft_re.real[5:10, :].sum(axis=0)
    # outdata['imag_7th_harmonic'] = fft_re.imag[5:10, :].sum(axis=0)

    fft_re_mean_features_num = 5
    fft_mean_window_size = int(fft_re.shape[0]/fft_re_mean_features_num)
    fft_mean_f = lambda arr:[arr[(i*fft_mean_window_size):((i+1)*fft_mean_window_size)].mean() \
                             for i in range(fft_re_mean_features_num)]

    fft_mean_re = np.apply_along_axis(fft_mean_f, axis=0, arr=fft_re_abs)
    for i in range(fft_mean_re.shape[0]):
        outdata['val_'+str(i)+'_fft_mean'] = fft_mean_re[i, :]


    mean_features_num = 5
    mean_window_size = int(rawdata.shape[0]/mean_features_num)
    mean_f = lambda arr:[arr[(i*mean_window_size):((i+1)*mean_window_size)].mean() \
                             for i in range(mean_features_num)]

    mean_re = np.apply_along_axis(mean_f, axis=0, arr=rawdata.values)
    for i in range(mean_re.shape[0]):
        outdata['val_'+str(i)+'_raw_mean'] = mean_re[i, :]

    return outdata


def readRawSignal_extractFeatures(path, subset_size=500, start_id=0, end_id=29049):
    relist = []

    processFun = lambda x: extractFeatures(
                pq.read_pandas(
                    path,
                    columns=[str(val) for val in range(start_id+x*subset_size, min(start_id+(x+1)*subset_size, end_id))]).to_pandas()
                )
    multiProcess = mutiProcessLoop(processFun, range(math.ceil((end_id-start_id)/subset_size)), n_process=5, silence=False)
    resultlist = multiProcess.run()
    return pd.concat(resultlist)

train_features = readRawSignal_extractFeatures('./rawdata/train.parquet',subset_size=60, start_id=0, end_id=8712)
test_features = readRawSignal_extractFeatures('./rawdata/test.parquet',subset_size=60, start_id=8712, end_id=29049)

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

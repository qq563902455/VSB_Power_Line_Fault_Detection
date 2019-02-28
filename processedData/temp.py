import pandas as pd
import pyarrow.parquet as pq
import os, psutil, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import math

from scipy.signal import spectrogram
from scipy.signal import welch
from lxyTools.process import mutiProcessLoop


def extractFeatures(rawdata):

    trend = rawdata.rolling(20000, center=True, min_periods=1, axis=0).mean()
    res = rawdata - trend

    res_std = res.std(axis=0)
    res_mean = res.mean(axis=0)
    res_abs = (res - res_mean).abs()

    outdata = pd.DataFrame()
    outdata['signal_id'] = rawdata.columns.values.astype(int)

    outdata['max_val_trend'] = trend.max(axis=0).values
    outdata['min_val_trend'] = trend.min(axis=0).values
    outdata['mean_val_trend'] = trend.mean(axis=0).values

    outdata['max_val_res'] = res.max(axis=0).values
    outdata['min_val_res'] = res.min(axis=0).values
    outdata['mean_val_res'] = res_mean.values

    outdata['val>2.5std_res'] = (res_abs>(2.5*res_std)).sum(axis=0).values
    outdata['val>5std_res'] = (res_abs>(5*res_std)).sum(axis=0).values
    outdata['val>10std_res'] = (res_abs>(10*res_std)).sum(axis=0).values
    outdata['val>20std_res'] = (res_abs>(20*res_std)).sum(axis=0).values

    diff_1_rawdata = np.diff(rawdata.values, n=1, axis=0)
    diff_2_rawdata = np.diff(rawdata.values, n=2, axis=0)

    outdata['max_val_diff1'] = diff_1_rawdata.max(axis=0)
    outdata['min_val_diff1'] = diff_1_rawdata.min(axis=0)
    outdata['std_val_diff1'] = diff_1_rawdata.std(axis=0)
    outdata['mean_val_diff1'] = diff_1_rawdata.mean(axis=0)

    outdata['max_val_diff2'] = diff_2_rawdata.max(axis=0)
    outdata['min_val_diff2'] = diff_2_rawdata.min(axis=0)
    outdata['std_val_diff2'] = diff_2_rawdata.std(axis=0)
    outdata['mean_val_diff2'] = diff_2_rawdata.mean(axis=0)

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
    multiProcess = mutiProcessLoop(processFun, range(math.ceil((end_id-start_id)/subset_size)), n_process=4, silence=False)
    resultlist = multiProcess.run()
    return pd.concat(resultlist)

result = readRawSignal_extractFeatures('./rawdata/train.parquet',subset_size=50, start_id=0, end_id=8712)


rawdata =pq.read_pandas('./rawdata/train.parquet',
           columns=[str(val) for val in range(100)]).to_pandas()
rawdata.shape

temp.reshape(-1).shape
f, Pxx = welch(rawdata.values, nperseg=1024, noverlap=256, axis=0)
outdata['welch_rawdata>2.5'] = (Pxx>2.5).sum(axis=0)
outdata['welch_rawdata>80'] = (Pxx>80).sum(axis=0)

search_f = lambda arr: f[np.where(arr == arr.max())]

temp = np.apply_along_axis(search_f, axis=0, arr=Pxx)

relist[0].head()



len(train_features)
start = time.time()
temp = train_features[0].rolling(20000, center=True, min_periods=1, axis=0).mean()
end =  time.time()
print(end-start)
train_features[0].shape

start = time.time()
for i in range(20):
    ans = (train_features[0] - temp)
end =  time.time()
print(end-start)


f, t, Sxx = spectrogram(train_features[0].values, nperseg=32, noverlap=8, axis=0)

Sxx.shape

Pxx == Pxx.max(axis=0)

np.where(Pxx == Pxx.max(axis=0))[0]
f.shape
Pxx
Pxx.max(axis=0)

np.where(Pxx == Pxx.max(axis=0))

Pxx.shape
f.shape
outDataFrame.signal_id

np.diff(temp.values, n=1, axis=0).shape

temp.abs()>temp.std(axis=0)

outDataFrame = pd.DataFrame()
outDataFrame['signal_id'] = temp.columns.values.astype(int)
outDataFrame['max'] = temp.max(axis=0)
temp.max(axis=0)
outDataFrame

type(temp.max(axis=0))
temp.max(axis=0)
process = psutil.Process(os.getpid())
print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')

re.shape
pq.read_pandas(
    path,
    columns=[str(val) for val in range(start_id+x*subset_size, min(start_id+(x+1)*subset_size, end_id))]).to_pandas(),

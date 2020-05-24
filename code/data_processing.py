from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import MaxPooling1D
from keras.regularizers import l2
from keras.layers import BatchNormalization
from math import sqrt
from matplotlib import pyplot
from numpy import split
from numpy import array
from numpy.random import seed
from pandas import read_csv
from scipy.stats import normaltest
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tensorflow as tf

# Interpolate the middle point use stochastic interpolation built by last group
def interpolate_gaussian(data, num_between, var=8):  # Data should be univariate time series in pd dataframe

    new_dat = np.zeros(1)

    for i in range(len(data)):
        new_dat = np.append(new_dat, data.iloc[i, :].values[0])
        if (i != (len(data) - 1)):
            step = (data.iloc[i + 1, :].values[0] - data.iloc[i, :].values[0]) / (num_between + 1)
            for j in range(num_between):
                k = data.iloc[i, :].values[0] + j * step + np.random.normal(loc=0.0,
                                                                            scale=1 / (var * (num_between + 1)))
                new_dat = np.append(new_dat, k)
    interpolated_data = pd.DataFrame(new_dat)
    interpolated_data = interpolated_data.iloc[1:len(interpolated_data), :]

    return (interpolated_data)


def prep_dat(d, n_step=270, n_ahead=360):  # Train a network for every step ahead (4 quarters, 8 quarters)
    # Prepare data for input/output of NNs, n_step is number of historical time steps (days) to include in regression task
    # to predict next quarter cash flows, assuming 90 days in a quarter

    a = d.copy()

    for i in range(1, n_step):
        ind = "t+" + str(i)
        a[ind] = a.iloc[:, 0].shift(-i)  # Use n_step historical observations to predict next quarter

    a["Q_Next"] = a.iloc[:, 0].shift(-(n_step + n_ahead))  # Next quarter(s) prediction

    a.dropna(
        inplace=True)  # last (n_step + n_ahead) rows will be NANs since shifting forward for "Next Quarter" Variable

    return (a)

#get distribution
def get_main(files, sheet):
    d = pd.read_excel(files, sheet_name = sheet, header = 2, usecols = "A:G",index_col = 0)[['Distributed']]
    d.dropna()
    delta = data_process(d)
    delta = delta.rename(columns = {0:sheet})
    return delta

def Pca(data,num_com):
    testy = data["All"]
    testx = data.iloc[:,1:]
    pca = PCA(n_components=num_com)
    fit = pca.fit_transform(testx)
    ratio = pca.fit(testx)
    print(f"total explained variance for {num_com} components is",sum(ratio.explained_variance_ratio_))
    dataset2 = pd.concat([testy.reset_index().drop(columns = 'index'),pd.DataFrame(fit)],axis = 1)
    return dataset2


def data_process(data):

    log_data = np.log(data).diff().dropna()
    _train = interpolate_gaussian(log_data.iloc[0:math.floor(0.7*len(log_data))],in_between,6)
    _test = interpolate_gaussian(log_data.iloc[math.floor(0.7*len(log_data)):len(log_data)],in_between,8)
    result = pd.concat([_train, _test])
    return result 

def get_macro(files, sheet):
    d = pd.read_excel(files, sheet_name = sheet, header = 2, usecols = "A:G",index_col = 0)[["Called Up"]]
    d = d.dropna()
    delta = data_process(d)
    delta = delta.rename(columns = {0:sheet})
    return delta

# take pd dataframe into processable numpy array
def trans_dat(data):
    pre_col = 0
    for col in data.columns:
        if pre_col == 0:
            sup_dat = prep_dat(data[[col]]).drop(columns='Q_Next').values
            sup_dat = np.expand_dims(sup_dat,axis=2)
            pre_col = col
        else:
            
            sup_dat1 = prep_dat(data[[col]]).drop(columns='Q_Next').values
            sup_dat1 = np.expand_dims(sup_dat1,axis=2)
            sup_dat = np.concatenate((sup_dat,sup_dat1),axis=2)            
            
    return sup_dat

def split_dataset_x(data):
    # split into standard weeks
    num_features = data.shape[2]
    train =  data[0:math.floor(0.7 * len(data)),0:hidden_size,0:num_features]
    test = data[math.floor(0.7 * len(data)):-1,0:hidden_size,0:num_features]
#     # restructure into windows of weekly data
#     train = array(split(train, int(len(train)/7)))
#     test = array(split(test, int(len(test)/7)))
    return train, test
# 1176/7 = 168, 546/7 = 78
def split_dataset_y(data):
    train = np.expand_dims(data[0:math.floor(0.7 * len(data))],axis = 1)
    test = np.expand_dims(data[math.floor(0.7 * len(data)):-1],axis = 1)
    return train,test



filename = "Buyout_Funds_Stats_2014.xlsx"
metric = 'Called Up'
in_between = 90
#The time used to predict the future, equal to n_step in prep_dat
hidden_size = 270
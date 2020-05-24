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

def evaluation_model(model,X_train,X_test,y_train,y_test):
    import matplotlib.pyplot as plt
    # %matplotlib inline
    history = model.history
    

    y_hat_train = model.predict(X_train,verbose = 0)
    y_hat_test = model.predict(X_test,verbose = 0)
    train_mse = mean_squared_error(y_hat_train[:,0],y_train)
    test_mse = mean_squared_error(y_hat_test[:,0],y_test)
    
    print("Train set MSE is",train_mse)
    print("Test set MSE is",test_mse)
    
    plt.figure(figsize=(6, 3))
    plt.plot(history.history['loss'],label = 'training loss')
#     plt.plot(history.history['val_loss'],label = 'validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Training loss')
    plt.legend()
    
    
    plt.figure(figsize=(6, 3))
    plt.plot(y_hat_train[:,0],label = 'prediction')
    plt.plot(y_train,label = 'actual')
    plt.ylabel('value')
    plt.xlabel('data point')
    plt.title('Train set actual/predict')
    plt.legend()
    
    
    plt.figure(figsize=(6, 3))
    plt.plot(y_hat_test[:,0],label = 'prediction')
    plt.plot(y_test,label = 'actual')
    plt.ylabel('value')
    plt.xlabel('data point')
    plt.suptitle('Test set actual/predict')
    plt.legend()

def evaluation_model_conv(model,X_train,X_test,y_train,y_test,n_steps=3,n_length=90):
    import matplotlib.pyplot as plt
    # %matplotlib inline
    history = model.history
    
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], n_steps, 1, n_length, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
    y_hat_train = model.predict(X_train,verbose = 0)
    y_hat_test = model.predict(X_test,verbose = 0)
    
    train_mse = mean_squared_error(y_hat_train[:,0],y_train)
    test_mse = mean_squared_error(y_hat_test[:,0],y_test)
    
    print("Train set MSE is",train_mse)
    print("Test set MSE is",test_mse)
    
    plt.figure(figsize=(6, 3))
    plt.plot(history.history['loss'],label = 'training loss')
#     plt.plot(history.history['val_loss'],label = 'validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Training loss')
    plt.legend()
    
    
    plt.figure(figsize=(6, 3))
    plt.plot(y_hat_train[:,0],label = 'prediction')
    plt.plot(y_train,label = 'actual')
    plt.ylabel('value')
    plt.xlabel('data point')
    plt.title('Train set actual/predict')
    plt.legend()
    
    
    plt.figure(figsize=(6, 3))
    plt.plot(y_hat_test[:,0],label = 'prediction')
    plt.plot(y_test,label = 'actual')
    plt.ylabel('value')
    plt.xlabel('data point')
    plt.suptitle('Test set actual/predict')
    plt.legend()

n_steps = 3
n_length = 90
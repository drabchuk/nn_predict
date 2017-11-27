import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import time
import excel_reader
from neuralnet import NeuralNet

DATA_FROM = 0
DATA_TO = 300000
DATA_LENGTH = DATA_TO - DATA_FROM
INF_CANDLES = 90


def linear_regression(x, y):
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    print('Coef: ', regr.coef_)
    return regr.coef_, regr.intercept_


def std_moving_average(p, log_price, i, period):
    return np.mean(log_price[i - period:i - 1, :])


def feature_transform(p, log_price, i):
    raw = log_price[i - 5:i - 1, :]
    ma_10 = std_moving_average(p, log_price, i, 10)
    ma_20 = std_moving_average(p, log_price, i, 20)
    ma_40 = std_moving_average(p, log_price, i, 40)
    ma_80 = std_moving_average(p, log_price, i, 80)
    ma_160 = std_moving_average(p, log_price, i, 160)
    ma_320 = std_moving_average(p, log_price, i, 320)
    ma_640 = std_moving_average(p, log_price, i, 640)
    features = np.vstack((raw,
                         ma_10,
                         ma_20,
                         ma_40,
                         ma_80,
                         ma_160,
                         ma_320,
                         ma_640,
                          ))
    features = (features - log_price[i, 0]) * 200.0
    return features


def label_data():
    p = excel_reader.get_data(DATA_FROM, DATA_TO, 'D:\python\projdata\data\\1m.xlsx')
    log_price = np.log(p)
    plt.plot(p)
    plt.show()
    plt.plot(log_price)
    minutes1d = range(DATA_LENGTH)
    minutes = np.array([minutes1d]).reshape(-1, 1)
    am, bm = linear_regression(minutes, log_price)
    a = am[0, 0]
    b = bm[0]
    minute_line = range(DATA_LENGTH)
    minute_line = np.array([minutes1d]).reshape(-1, 1)
    line = minute_line * a + b
    plt.plot(line)
    plt.show()
    lin_reg = log_price - line
    plt.plot(lin_reg)
    plt.show()

label_data()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import time
import excel_reader
from neuralnet import NeuralNet

DATA_FROM = 0
DATA_TO = 349000
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

def exp_conv_array(d, alfa):
    conv = np.ones((d,), dtype=float)
    a = alfa
    for i in range(10):
        conv[i] = a
        a *= a
    conv = conv / np.sum(conv)
    return conv


def convolutions_long(x):
    alfa = 0.9
    #x = x[:, 0]
    conv1280 = exp_conv_array(1280, 0.9)
    conv2560 = exp_conv_array(2560, 0.9)
    conv5120 = exp_conv_array(5120, 0.9)
    conv10240 = exp_conv_array(10240, 0.9)
    conv20480 = exp_conv_array(20480, 0.9)
    conv40960 = exp_conv_array(40960, 0.9)
    ma10240 = np.convolve(x, conv10240, 'same')
    #ma20480 = np.convolve(x, conv20480, 'same')
    #ma40960 = np.convolve(x, conv40960, 'same')
    plt.plot(x)
    plt.plot(ma10240)
    #plt.plot(ma20480)
    #plt.plot(ma40960)
    plt.show()




def convolutions(x):
    alfa = 0.9
    #x = x[:, 0]
    conv10 = exp_conv_array(10, 0.9)
    conv20 = exp_conv_array(20, 0.9)
    conv40 = exp_conv_array(40, 0.9)
    conv80 = exp_conv_array(80, 0.9)
    conv160 = exp_conv_array(160, 0.9)
    conv320 = exp_conv_array(320, 0.9)
    conv640 = exp_conv_array(640, 0.9)
    conv1280 = exp_conv_array(1280, 0.9)
    conv2560 = exp_conv_array(2560, 0.9)
    conv5120 = exp_conv_array(5120, 0.9)
    conv10240 = exp_conv_array(10240, 0.9)
    conv20480 = exp_conv_array(20480, 0.9)
    conv40960 = exp_conv_array(40960, 0.9)
    ma10 = np.convolve(x, conv10, 'same')
    ma20 = np.convolve(x, conv20, 'same')
    ma40 = np.convolve(x, conv40, 'same')
    ma80 = np.convolve(x, conv80, 'same')
    ma160 = np.convolve(x, conv160, 'same')
    ma320 = np.convolve(x, conv320, 'same')
    ma640 = np.convolve(x, conv640, 'same')
    ma1280 = np.convolve(x, conv1280, 'same')
    ma2560 = np.convolve(x, conv2560, 'same')
    ma5120 = np.convolve(x, conv5120, 'same')
    ma10240 = np.convolve(x, conv10240, 'same')
    ma20480 = np.convolve(x, conv20480, 'same')
    ma40960 = np.convolve(x, conv40960, 'same')
    plt.plot(x)
    plt.plot(ma10)
    plt.plot(ma20)
    plt.plot(ma40)
    plt.plot(ma80)
    plt.plot(ma160)
    plt.plot(ma320)
    plt.plot(ma640)
    plt.plot(ma1280)
    plt.plot(ma2560)
    plt.plot(ma5120)
    plt.plot(ma10240)
    plt.plot(ma20480)
    plt.plot(ma40960)
    plt.show()

def convolutions_short(x):
    alfa = 0.9
    #x = x[:, 0]
    conv640 = exp_conv_array(640, 0.9)
    conv1280 = exp_conv_array(1280, 0.9)
    conv2560 = exp_conv_array(2560, 0.9)
    conv5120 = exp_conv_array(5120, 0.9)


    ma640 = np.convolve(x, conv640, 'same')
    ma1280 = np.convolve(x, conv1280, 'same')
    ma2560 = np.convolve(x, conv2560, 'same')
    ma5120 = np.convolve(x, conv5120, 'same')
    plt.plot(x)
    plt.plot(ma640)
    plt.plot(ma1280)
    plt.plot(ma2560)
    plt.plot(ma5120)
    plt.show()

def label_data():
    p = excel_reader.get_data(DATA_FROM, DATA_TO, 'D:\python\projdata\data\\1m.xlsx')
    log_price = np.log(p)
    #plt.plot(p)
    #plt.show()
    #plt.plot(log_price)
    minutes1d = range(DATA_LENGTH)
    minutes = np.array([minutes1d]).reshape(-1, 1)
    am, bm = linear_regression(minutes, log_price)
    a = am[0, 0]
    b = bm[0]
    minute_line = range(DATA_LENGTH)
    minute_line = np.array([minutes1d]).reshape(-1, 1)
    line = minute_line * a + b
    #plt.plot(line)
    #plt.show()
    lin_reg = log_price - line
    lin_reg = lin_reg[:, 0]
    plt.plot(lin_reg)
    #conv_array = np.ones((43200,), dtype=float)/43200.
    conv_array = np.ones((40960,), dtype=float)/40960.
    ma = np.convolve(lin_reg, conv_array, 'same')
    plt.plot(ma)
    plt.show()
    without_month_avg = lin_reg - ma
    plt.plot(without_month_avg)
    plt.show()
    convolutions_long(without_month_avg)
    conv_array_10k = np.ones((10240,), dtype=float) / 10240.
    ma10k = np.convolve(without_month_avg, conv_array_10k, 'same')
    without_10d_avg = without_month_avg - ma10k
    plt.plot(without_10d_avg)
    plt.show()
    convolutions_short(without_10d_avg)
    #plt.plot(ma)
    #convolutions(lin_reg)
    #min - 7000 - 0.16
    #min - 22000 - 0.16
    #0 - 38000
    #+ - 80000
    #0 - 118000
    #+ - 70000
    #0 - 188000
    #+ - 60000
    #0 - 248000
    #semi-period - 70000
    period = 125000.0
    amplitude = 0.18
    zero = 238000.0
    #trend_3m = amplitude * np.sin(((minute_line - zero) / period) * 2.0 * np.pi)
    #plt.plot(trend_3m)
    #plt.show()
    #without_trend_3m = lin_reg - trend_3m
    #plt.plot(without_trend_3m)
    #plt.show()



label_data()

import numpy as np
import matplotlib.pyplot as plt
import time
import excel_reader
import nn_factory
from neuralnet import NeuralNet

DATA_FROM = 0
DATA_TO = 100000
DATA_LENGTH = DATA_TO - DATA_FROM
INF_CANDLES = 90
#LOGNORMAL_MU = -2
#LOGNORMAL_STD = 0.5
#mean = 0.22
#LOGNORMAL_MU = -5
#LOGNORMAL_STD = 0.01
C = 1.0
LOG_C_D = 2. * np.log(C)
LEARNING_RATE = 100
STEPS = 1000


def comp_point_profit_buy(b, s, log_price, point, sell_day):
    #b[k](s[k + 1]*index[k + 1, k] + (1 - s[k + 1])*(s[2]*index[k+2][k]...))
    #calc form inside
    last_inf_day = point + sell_day
    accum = 0.0
    for i in range(last_inf_day, point, -1):
        accum *= (1.0 - s[i])
        accum += s[i] * (log_price[i] - log_price[point] + LOG_C_D)
    accum *= b[point]
    return accum


def comp_point_profit_sell(b, s, log_price, point, sell_day):
    #b[k](s[k + 1]*index[k + 1, k] + (1 - s[k + 1])*(s[2]*index[k+2][k]...))
    #calc form inside
    last_inf_day = point + sell_day
    accum = 0.0
    for i in range(last_inf_day, point, -1):
        accum *= (1.0 - b[i])
        accum += b[i] * (log_price[point] - log_price[i] + LOG_C_D)
    accum *= s[point]
    return accum


def comp_loss(b, s, log_price):
    loss_buy = np.zeros((DATA_LENGTH, 1), dtype=float)
    loss_sell = np.zeros((DATA_LENGTH, 1), dtype=float)
    for i in range(DATA_LENGTH - 640):
        loss_buy[i, 0] = comp_point_profit_buy(b, s, log_price, i, INF_CANDLES)
        loss_sell[i, 0] = comp_point_profit_sell(b, s, log_price, i, INF_CANDLES)
    #plt.plot(loss_buy)
    #plt.plot(loss_sell)
    return np.sum(loss_buy) + np.sum(loss_sell)


def grad_b_s(b, s, log_price):
    db = np.zeros((DATA_LENGTH, 1), dtype=float)
    ds = np.zeros((DATA_LENGTH, 1), dtype=float)
    for i in range(INF_CANDLES, DATA_LENGTH - INF_CANDLES - 1):
        sum_db = 0.0
        sum_db += comp_point_profit_buy(b, s, log_price, i, INF_CANDLES) / b[i]
        hold_prob = 1.0
        for j in range(i - 1, i - INF_CANDLES - 1, -1):
            future_buy_profit = comp_point_profit_sell(b, s, log_price, i, i - j) / s[i]
            local = log_price[j] - log_price[i] - future_buy_profit
            past_influence = s[j] * local * hold_prob
            #print('buy past inf for', i, ' on ', j, ' is ', past_influence)
            sum_db += past_influence
            hold_prob *= (1 - b[j])
        db[i] = sum_db
    for i in range(INF_CANDLES, DATA_LENGTH - INF_CANDLES - 1):
        sum_ds = 0.0
        sum_ds += comp_point_profit_sell(b, s, log_price, i, INF_CANDLES) / s[i]
        hold_prob = 1.0
        for j in range(i, i - INF_CANDLES - 1, -1):
            #print('infl', j)
            future_sell_profit = comp_point_profit_buy(b, s, log_price, i, i - j) / b[i]
            local = log_price[i] - log_price[j] - future_sell_profit
            past_influence = b[j] * local * hold_prob
            #print('sell past inf for', i, ' on ', j, ' is ', past_influence)
            sum_ds += past_influence
            hold_prob *= (1 - s[j])
        ds[i] = sum_ds
    return db, ds


def init_b_s(p):
    b = np.zeros((DATA_LENGTH, 1), dtype=float)
    s = np.zeros((DATA_LENGTH, 1), dtype=float)
    for i in range(DATA_LENGTH - 1):
        if p[i + 1] > p[i]:
            b[i] = 1.0
            s[i] = 0.0
        else:
            s[i] = 1.0
            b[i] = 0.0

    return b, s

def comp_b_s(nn, p, log_price):
    b = np.zeros((DATA_LENGTH, 1), dtype=float)
    s = np.zeros((DATA_LENGTH, 1), dtype=float)
    inf_prices = nn.topology[0]
    for i in range(640, DATA_LENGTH):
        #features = (log_price[i - inf_prices:i] - log_price[i]) * 20
        features = feature_transform(p, log_price, i)
        a = nn.forward(features)
        b[i] = a[0, 0]
        s[i] = a[1, 0]
    return b, s

def gen_theta_acc(topology):
    acc = []
    for i in range(len(topology) - 1):
        acc.append(np.zeros((topology[i] + 1, topology[i + 1])))
    return acc


def sum(a, b):
    for i in range(len(a)):
        a[i] = a[i] + b[i]
    return a


def div(a, b):
    for i in range(len(a)):
        a[i] = a[i] / b
    return a

def mul(a, b):
    for i in range(len(a)):
        a[i] = a[i] * b
    return a

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

def gradient_descent(nn, p, log_price, steps, learning_rate):
    print('gradient descent: ')
    print('steps: ', steps)
    print('learning rate: ', learning_rate)
    inf_prices = nn.topology[0]
    m = DATA_LENGTH - 2 * INF_CANDLES
    for step in range(steps):
        tic = time.clock()
        b, s = comp_b_s(nn, p, log_price)
        toc = time.clock()
        print('b s compute took ', toc - tic)
        print('step', step)
        tic = time.clock()
        loss = comp_loss(b, s, log_price)
        toc = time.clock()
        print('compute loss took ', toc - tic)
        print('loss', loss)
        tic = time.clock()
        db, ds = grad_b_s(b, s, log_price)
        toc = time.clock()
        print('compute grad took ', toc - tic)
        acc_theta_grad = gen_theta_acc(nn.topology)
        tic = time.clock()
        for sample in range(640, DATA_LENGTH - INF_CANDLES):
            #features = (log_price[sample - inf_prices:sample, :] - log_price[sample, :]) * 20
            features = feature_transform(p, log_price, sample)
            a = nn.forward(features)
            delta_out = np.vstack((db[sample, 0], ds[sample, 0]))
            grad = nn.backprop(delta_out)
            acc_theta_grad = sum(acc_theta_grad, grad)
        toc = time.clock()
        print('backprop took ', toc - tic)
        weighted_grad = div(acc_theta_grad, DATA_LENGTH - 640 - INF_CANDLES)
        weight_update = mul(weighted_grad, LEARNING_RATE)
        nn.theta = sum(nn.theta, weight_update)
        nn.save('net_without_comm')

        #b = b + (db * learning_rate)
        #s = s + (ds * learning_rate)
    return b, s


def label_data():
    p = excel_reader.get_data(DATA_FROM, DATA_TO, 'D:\python\projdata\data\\1m.xlsx')
    log_price = np.log(p)
    #plt.plot(p)
    topology = [11, 100, 100, 50, 20, 2]
    #nn = NeuralNet(topology)
    nn = nn_factory.read('net_11_7d')
    #index = comp_index_matrix(p)
    #b, s = comp_b_s(nn, p, index)
    #plt.plot(index[0,:])
    #comp_loss(b, s, index)
    #plt.plot(p / 4000 - 1)
    #db, ds = grad_b_s(b, s, index)
    #plt.plot(ds * 10)
    #plt.show()
    lb, ls = gradient_descent(nn, p, log_price, STEPS, LEARNING_RATE)
    plt.plot(lb)
    plt.show()
    nn.save('net_final')

label_data()

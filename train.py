import numpy as np
import matplotlib.pyplot as plt
import time
import excel_reader

DATA_FROM = 0
DATA_TO = 300
DATA_LENGTH = DATA_TO - DATA_FROM
INF_DAYS = 50

LOGNORMAL_MU = -5
LOGNORMAL_STD = 0.01
C = 0.999
LEARNING_RATE = 0.1
STEPS = 10

def comp_index_matrix(p):
    data_length = p.shape[0]
    alfa = np.dot(p, (1.0 / p.T))
    alfa = np.log(alfa)
    alfa += 2.0 * np.log(C)
    return alfa


def comp_point_profit_buy(b, s, index, point, sell_day):
    #b[k](s[k + 1]*index[k + 1, k] + (1 - s[k + 1])*(s[2]*index[k+2][k]...))
    #calc form inside
    last_inf_day = point + sell_day
    accum = 0.0
    for i in range(last_inf_day, point, -1):
        accum *= (1.0 - s[i])
        accum += s[i] * index[i, point]
    accum *= b[point]
    return accum


def comp_point_profit_sell(b, s, index, point, sell_day):
    #b[k](s[k + 1]*index[k + 1, k] + (1 - s[k + 1])*(s[2]*index[k+2][k]...))
    #calc form inside
    last_inf_day = point + sell_day
    accum = 0.0
    for i in range(last_inf_day, point, -1):
        accum *= (1.0 - b[i])
        accum += b[i] * index[point, i]
    accum *= s[point]
    return accum


def comp_loss(b, s, index):
    loss_buy = np.zeros((DATA_LENGTH, 1), dtype=float)
    loss_sell = np.zeros((DATA_LENGTH, 1), dtype=float)
    for i in range(DATA_LENGTH - INF_DAYS):
        loss_buy[i, 0] = comp_point_profit_buy(b, s, index, i, INF_DAYS)
        loss_sell[i, 0] = comp_point_profit_sell(b, s, index, i, INF_DAYS)
    #plt.plot(loss_buy)
    #plt.plot(loss_sell)
    return sum(loss_buy) + sum(loss_sell)


def grad_b_s(b, s, index):
    db = np.zeros((DATA_LENGTH, 1), dtype=float)
    ds = np.zeros((DATA_LENGTH, 1), dtype=float)
    for i in range(INF_DAYS, DATA_LENGTH - INF_DAYS - 1):
        sum_db = 0.0
        sum_db += comp_point_profit_buy(b, s, index, i, INF_DAYS) / b[i]
        hold_prob = 1.0
        for j in range(i - 1, i - INF_DAYS - 1, -1):
            future_buy_profit = comp_point_profit_sell(b, s, index, i, i - j) / s[i]
            local = index[j, i] - future_buy_profit
            past_influence = s[j] * local * hold_prob
            #print('buy past inf for', i, ' on ', j, ' is ', past_influence)
            sum_db += past_influence
            hold_prob *= (1 - b[j])
        db[i] = sum_db
    for i in range(INF_DAYS, DATA_LENGTH - INF_DAYS - 1):
        sum_ds = 0.0
        sum_ds += comp_point_profit_sell(b, s, index, i, INF_DAYS) / s[i]
        hold_prob = 1.0
        for j in range(i, i - INF_DAYS - 1, -1):
            #print('infl', j)
            future_sell_profit = comp_point_profit_buy(b, s, index, i, i - j) / b[i]
            local = index[i, j] - future_sell_profit
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


def init_b_s_lognormal(p):
    b = np.random.lognormal(LOGNORMAL_MU, LOGNORMAL_STD, (DATA_LENGTH, 1))
    s = np.random.lognormal(LOGNORMAL_MU, LOGNORMAL_STD, (DATA_LENGTH, 1))
    return b, s

def gradient_descent(b, s, index, steps, learning_rate):
    print('gradient descent: ')
    print('steps: ', steps)
    print('learning rate: ', learning_rate)
    for step in range(steps):
        print('step', step)
        loss = comp_loss(b, s, index)
        print('loss', loss)
        db, ds = grad_b_s(b, s, index)
        b = b + (db * learning_rate)
        s = s + (ds * learning_rate)
    return b, s


def label_data():
    p = excel_reader.get_data(DATA_FROM, DATA_TO)
    np.random.seed(int(time.time()))
    #plt.plot(p)
    b, s = init_b_s_lognormal(p)
    index = comp_index_matrix(p)
    comp_loss(b, s, index)
    plt.plot(p / 4000 - 1)
    db, ds = grad_b_s(b, s, index)
    #plt.plot(ds * 10)
    #plt.show()
    lb, ls = gradient_descent(b, s, index, STEPS, LEARNING_RATE)
    plt.plot(lb)
    plt.show()

label_data()

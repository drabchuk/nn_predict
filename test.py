import numpy as np
import matplotlib.pyplot as plt
import nn_factory
import excel_reader
import random
#C = 0.999
C = 1.0

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

DATA_FROM, DATA_TO = 0, 100000
TRAINING_LENGTH = DATA_TO - DATA_FROM
nn = nn_factory.read('D://python/TorgovecNets/net_w_c_loss_-40')
#p = excel_reader.get_data(DATA_FROM, DATA_TO, 'D:\python\projdata\data\\btc30m.xlsx')
p = excel_reader.get_data(DATA_FROM, DATA_TO, 'D:\python\projdata\data\\1m_short.xlsx')
#index = comp_index_matrix(p)
log_price = np.log(p)
inf_prices = 100
print('TEST')
wallet_btc = 1.0
wallet_usd = 0.0
sell_flag = True
trades_btc = 0
trades_usd = 0
activations = np.zeros((TRAINING_LENGTH, 2))
hold_btc = 0
hold_usd = 0
sell_btc = np.zeros((TRAINING_LENGTH, 1))
buy_btc = np.zeros((TRAINING_LENGTH, 1))
#treshhold = random.random()
treshhold_buy = 0.994
treshhold_sell = 0.915
#treshhold_buy = 0.8
#treshhold_sell = 0.8
upper_bound = 0.3
lower_bound = 0.2
treshhold = 0.25
price_buy = 1.0
price_sell = 0.0
for sample in range(640, TRAINING_LENGTH - inf_prices):
    #features = index[sample - inf_prices:sample, sample] * 20
    features = feature_transform(p, log_price, sample)
    a = nn.forward(features)
    a_inv = nn.forward(-features)
    buy = a[0, 0]
    #sell = a_inv[0, 0]
    sell = a[1, 0]
    activations[sample, 0] = buy
    activations[sample, 1] = sell

    if sell_flag:
        #if buy < lower_bound:
        if sell > treshhold_sell:
        #if buy > treshhold_buy:
        #if buy > treshhold_buy or p[sample] / price_sell < 0.99:
            price_buy = p[sample]
            print('sell on ', sample, ' price: ', p[sample])
            sell_btc[sample, 0] = 1
            trades_btc += 1
            wallet_usd = wallet_btc * p[sample] * C
            print('USD', wallet_usd)
            wallet_btc = 0.0
            sell_flag = False
        else:
            hold_btc += 1
    else:
        #if buy > upper_bound:
        if buy > treshhold_buy:
        #if sell > treshhold_sell:
        #if sell > treshhold_sell or p[sample] / price_buy > 1.01:
            price_sell = p[sample]
            print('by on ', sample, ' price: ', p[sample])
            buy_btc[sample, 0] = 1
            trades_usd += 1
            wallet_btc = wallet_usd / p[sample] * C
            print('BTC', wallet_btc)
            wallet_usd = 0.0
            sell_flag = True
        else:
            hold_usd += 1

print('btc ', wallet_btc)
print('usd ', wallet_usd)
print('trades btc', trades_btc, '/', TRAINING_LENGTH)
print('trades usd', trades_usd, '/', TRAINING_LENGTH)


print('hold_btc: ', hold_btc)
print('hold_usd: ', hold_usd)
plt.hist(activations[:, 0], 1000)
plt.show()
plt.hist(activations[:, 1], 1000)
plt.show()

price_g = (p - np.mean(p)) / np.std(p) / 10 + 0.3
plt.plot(price_g)
#plt.plot(activations)
plt.plot(sell_btc)
plt.plot(buy_btc)
plt.show()
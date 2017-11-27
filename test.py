import numpy as np
import matplotlib.pyplot as plt
import nn_factory
import excel_reader
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

DATA_FROM, DATA_TO = 0, 5000
TRAINING_LENGTH = DATA_TO - DATA_FROM
nn = nn_factory.read('net_7d')
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
for sample in range(640, TRAINING_LENGTH - inf_prices):
    #features = index[sample - inf_prices:sample, sample] * 20
    features = feature_transform(p, log_price, sample)
    a = nn.forward(features)
    buy = a[0, 0]
    sell = a[1, 0]
    activations[sample, 0] = buy
    activations[sample, 1] = sell
    if sell_flag:
        if sell > 0.5:
            trades_btc += 1
            wallet_usd = wallet_btc * p[sample] * C
            print('USD', wallet_usd)
            wallet_btc = 0.0
            sell_flag = False
    else:
        if buy > 0.5:
            trades_usd += 1
            wallet_btc = wallet_usd / p[sample] * C
            print('BTC', wallet_btc)
            wallet_usd = 0.0
            sell_flag = True

print('btc ', wallet_btc)
print('usd ', wallet_usd)
print('trades btc', trades_btc, '/', TRAINING_LENGTH)
print('trades usd', trades_usd, '/', TRAINING_LENGTH)
price_g = (p - np.mean(p)) / np.std(p) / 50
plt.plot(price_g)
plt.plot(activations)
plt.show()
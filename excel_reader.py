import numpy as np
import pandas as pd
import time


def get_data(from_num, to_num, file_name):
    print('excel read')
    tic = time.clock()
    #raw_data = np.array(pd.read_excel('D:\python\projdata\data\\1mshort.xlsx'))
    raw_data = np.array(pd.read_excel(file_name))
    toc = time.clock()
    print('file read took ', toc - tic)
    price = raw_data[from_num:to_num, 0]
    tic = time.clock()
    price = price.reshape(to_num - from_num, 1)
    toc = time.clock()
    print('reshape took ', toc - tic)
    return price

##### test day

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math
import openpyxl
from itertools import zip_longest
from operator import is_not
from functools import partial


list_of_etfs = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
date_range = pd.date_range(start="1992-01-01", end="2020-12-31", freq="D")
data = []
for index in range(len(list_of_etfs)):
    part = yf.download(list_of_etfs[index], start = '1992-01-01', end = '2020-12-31')['Adj Close']
    data.append(part)
    
data = pd.concat(data, axis = 1)
data.dropna()
new = data.pct_change().dropna()

new.to_csv("C:/Users/rycba/OneDrive/Documents/testetf.csv")

etf_log = np.log(data.dropna() / data.dropna().shift(1)).dropna()

etf_log.to_csv("C:/Users/rycba/OneDrive/Documents/etf_logreturns.csv")


GSPC = yf.download('^GSPC', start = '1990-01-01', end = '2024-11-16')['Adj Close']

GSPC_returns = GSPC.pct_change().dropna()

GSPC_logreturns = np.log(GSPC / GSPC.shift(1)).dropna()

GSPC_returns.to_csv("C:/Users/rycba/OneDrive/Documents/GSPC_returns.csv")
GSPC_logreturns.to_csv("C:/Users/rycba/OneDrive/Documents/GSPC_logreturns.csv")

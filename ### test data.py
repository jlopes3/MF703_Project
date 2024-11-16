### test data 
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math
import openpyxl
from itertools import zip_longest
from operator import is_not
from functools import partial





### Election Day Formula
def election_day(year):
    date = pd.Timestamp(year=year, month=11, day=1)
    return date + pd.offsets.Week(weekday=1)

### Time Interval
num = int((2020-1992) / 4 + 1)
election_days = {1992 + 4 * i : election_day(1992 + 4 * i) for i in range(num)}
election_days.items()

ran = list(range(1992, 2021, 4))

tickers = [
    "SPY", 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 
    "AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "GOOG", "GOOGL", "BRK-B", "META", "JPM",
    "V", "WMT", "MA", "PG", "UNH", "XOM", "HD", "CVX", "ABBV", "MRK",
    "ADBE", "PEP", "CMCSA", "NFLX", "KO", "PFE", "T", "CRM", "AVGO", "MCD",
    "BAC", "ORCL", "INTC", "ABT", "LLY", "LIN", "COST", "AMD", "HON", "AMGN",
    "TMO", "DIS", "NKE", "MDT", "DHR", "TXN", "BMY", "PYPL", "GS", "RTX",
    "PM", "SBUX", "IBM", "PLD", "C", "LMT", "NOW", "ISRG", "MU", "CI",
    "MDLZ", "BLK", "GE", "CAT", "CSCO", "CVS", "MO", "ADP", "EL", "SPGI",
    "MMC", "AXP", "NOC", "COP", "EQIX", "TGT", "REGN", "USB", "INTU", "F",
    "GM", "KMB", "BK", "CHTR", "EXC", "AMT", "HCA", "EMR", "BDX", "PGR",
    "WFC", "DG", "ICE", "MAR", "SYK", "ITW", "STZ", "EOG", "SLB", "LRCX"
]

stock_prices = []


for index in range(len(tickers)):
    ticker = tickers[index]

    data_partial = []
    for i, j in election_days.items():
        data = yf.download(ticker, start=(j - pd.DateOffset(months=6)), end=(j + pd.DateOffset(days=14)))['Adj Close']
        data_partial.append(data)
    stock_prices.append(data_partial)
    
prices = stock_prices[0]
returns = []
i = 0
for prices in stock_prices:
    # Verify if the price list is not empty
    if prices:
        # Calculate returns for each series in the price list
        ticker_returns = [data.pct_change().dropna() for data in prices if not data.empty]
        returns.append(ticker_returns)
    else:
        returns.append([])
    print(i)
    i+=1


returns
######






######

plt.figure(12,6)
plt.plot(stock_data)
plt.show()

### 1a

stock_data.ffill().dropna()
#no major gaps in data, so I am simply forward filling the data

### 1b
daily_returns = stock_data.pct_change().dropna()

### 1c
cov_matrix = daily_returns.cov()

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(eigenvalues)

# They are all positive. None are negative. Incredible.

### 1d

weighted_eigenvalues = eigenvalues / eigenvalues.sum()

i = 0
percent_var = .5
while weighted_eigenvalues[0:i].sum() < percent_var:
    significant_eigenvalues_fifty = weighted_eigenvalues[0:i+1]
    i+=1

print(significant_eigenvalues_fifty)

i = 0
percent_var = .9
while weighted_eigenvalues[0:i].sum() < percent_var:
    significant_eigenvalues_ninety = weighted_eigenvalues[0:i+1]
    i+=1

print(significant_eigenvalues_ninety)

len(significant_eigenvalues_ninety)
len(significant_eigenvalues_fifty)

### It makes sense that fifty percent takes only 2 eigenvalues, but it shocked me that to get to ninety i needed over 50. This shows that while the top 50 or so percent is easily explained, it is a lot harder to explain the bottom 50. It took 49 PC's to explain 40% of the variance, which is not very good.


###2c
a = 1
c = np.array([1,.1])

G = np.zeros((2,100))
G[0,:] = 1
G[1,:17] = .1

GT = G.transpose()

C = cov_matrix.to_numpy()

C_inv = np.linalg.inv(C)

R = daily_returns.mean().to_numpy()

left = G @ C_inv @ GT
right = G @ C_inv @ R

l = np.linalg.solve(left, right - 2*a*c)

w = (C_inv @ (R - GT @ l)) / 2*a

w.sum()

eigenvectors[0]

type(daily_returns)
daily_returns_test = daily_returns.tz_localize(tz = None)
daily_returns_test.to_excel("C:/Users/rycba/OneDrive/Documents/testdata.xlsx")

returns_df.to_csv("C:/Users/rycba/OneDrive/Documents/testdata_etf.xlsx")

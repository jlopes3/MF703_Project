# 
# Title: 
# Name: Aiden Perkins
# Email address: ajp15@bu.edu
# Description: 
# 
#
from FutureClass import future
import pandas as pd

tickers = ["SI1","CL1","DX1","ES1","FF1","FF4","GC1","NG1","TY1"]
filenames = [ticker + '.xlsx' for ticker in tickers]

futures = {}
for filename in filenames:
    temp = future(filename)
    futures[temp.symbol] = temp

all_correlation_matrix = pd.DataFrame(index=tickers, columns=tickers)

# Overall Correlation Matrices
for i, future_i_name in enumerate(tickers):
    for j, future_j_name in enumerate(tickers):
        if i <= j:  # Compute only upper triangle (including diagonal)
            corr = futures[future_i_name].correlation(futures[future_j_name])
            all_correlation_matrix.loc[future_i_name, future_j_name] = corr
            all_correlation_matrix.loc[future_j_name, future_i_name] = corr

#Election time correlations
election_correlation_matrix = pd.DataFrame(index=tickers,columns=tickers)
for i, future_i_name in enumerate(tickers):
    for j, future_j_name in enumerate(tickers):
        if i <= j:  # Compute only upper triangle (including diagonal)
            corr = futures[future_i_name].correlation(futures[future_j_name],period='election')
            election_correlation_matrix.loc[future_i_name, future_j_name] = corr
            election_correlation_matrix.loc[future_j_name, future_i_name] = corr

#Non Election time
non_election_correlation_matrix = pd.DataFrame(index=tickers,columns=tickers)
for i, future_i_name in enumerate(tickers):
    for j, future_j_name in enumerate(tickers):
        if i <= j:  # Compute only upper triangle (including diagonal)
            corr = futures[future_i_name].correlation(futures[future_j_name],period='non_election')
            non_election_correlation_matrix.loc[future_i_name, future_j_name] = corr
            non_election_correlation_matrix.loc[future_j_name, future_i_name] = corr
#%%
#Variance F Tests
results = []
df = pd.DataFrame()
for item in futures:
    results.append(futures[item].election_var_F_test())
df[['F Stat','p value']] = pd.DataFrame(results,index= tickers)
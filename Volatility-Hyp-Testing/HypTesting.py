from scipy import stats
import pandas as pd
import numpy as np

def vol_difference_test(months_before, months_after, dataset):
    # Load historical data for ^GSPC
    data = pd.read_csv(dataset)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[(data.index >= '1974-01-01') & (data.index <= '2023-12-31')]

    # Calculate daily returns for ^GSPC
    data['Daily Return'] = data['Adj Close'].pct_change()
    data = data.dropna(subset=['Daily Return'])

    # Separate data into election periods and non-election periods
    election_dates = [pd.Timestamp("1976-11-02"), pd.Timestamp("1980-11-04"), pd.Timestamp("1984-11-06"), pd.Timestamp("1988-11-08"), pd.Timestamp("1992-11-03"), pd.Timestamp("1996-11-05"), pd.Timestamp("2000-11-07"), pd.Timestamp("2004-11-02"), pd.Timestamp("2008-11-04"), pd.Timestamp("2012-11-06"), pd.Timestamp("2016-11-08"), pd.Timestamp("2020-11-03")]
    election_periods = [(day - pd.DateOffset(months=months_before), day + pd.DateOffset(months=months_after)) for day in election_dates]
    data['In Election Cycle'] = data.index.to_series().apply(
        lambda date: any(start <= date <= end for start, end in election_periods)
    )

    election_data = data[data['In Election Cycle']]
    non_election_data = data[~data['In Election Cycle']]

    election_volatility = election_data['Daily Return'].std()
    non_election_volatility = non_election_data['Daily Return'].std()

    # Annualize the volatility assuming 252 trading days per year
    annualized_election_volatility = election_volatility * np.sqrt(252)
    annualized_non_election_volatility = non_election_volatility * np.sqrt(252)

    # Levene's Test to see if the variances of Daily Returns are equal
    lev_stat, p_value = stats.levene(election_data['Daily Return'], non_election_data['Daily Return'])

    return annualized_election_volatility, annualized_non_election_volatility, lev_stat, p_value



p_values_table = pd.DataFrame(index=range(1,13), columns=range(1,13))
for x in range(1,13):
    for y in range(1,13):
        _, _, _, p_value = vol_difference_test(x, y, "GSPC.csv")
        print("Election Period: " + str(x) + " months before, " + str(y) + " months after. P-value: " + str(p_value))
        p_values_table.at[y, x] = p_value



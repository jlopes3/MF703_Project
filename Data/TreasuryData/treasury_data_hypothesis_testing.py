
from scipy import stats
import pandas as pd
import numpy as np

def vol_difference_test_two(months_before, months_after, dataset):
    """ 
    This function analyzes daily returns in specified election and non-election periods,
    computes annualized volatilities for both, and performs Levene's test to assess 
    variance differences.

    Args:
        months_before (integer): the number of months before an election to start an election period
        months_after (integer): the number of months after an election to end an election period
        dataset (string): the filepath of the dataset
        
    Returns:
        annualized_election_volatility (float): the annualized volatility of election periods
        annualized_non_election_volatility (float): the annualized volatility of non-election periods
        lev_stat (float): the Levene statistic for the daily returns of the non-election periods and election periods
        p_value (float): the p-value from the Levene test
    """
    # Load historical data
    data = pd.read_csv(dataset)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[(data.index >= '1974-01-01') & (data.index <= '2023-12-31')]

    # Separate data into election periods and non-election periods
    election_dates = [pd.Timestamp("1976-11-02"), pd.Timestamp("1980-11-04"), pd.Timestamp("1984-11-06"), pd.Timestamp("1988-11-08"), pd.Timestamp("1992-11-03"), pd.Timestamp("1996-11-05"), pd.Timestamp("2000-11-07"), pd.Timestamp("2004-11-02"), pd.Timestamp("2008-11-04"), pd.Timestamp("2012-11-06"), pd.Timestamp("2016-11-08"), pd.Timestamp("2020-11-03")]
    election_periods = [(day - pd.DateOffset(months=months_before), day + pd.DateOffset(months=months_after)) for day in election_dates]
    data['In Election Cycle'] = data.index.to_series().apply(
        lambda date: any(start <= date <= end for start, end in election_periods)
    )
    election_data = data[data['In Election Cycle']]
    non_election_data = data[~data['In Election Cycle']]

    # Calculate volatility
    election_volatility = []
    non_election_volatility = []
    annualized_election_volatility = []
    annualized_non_election_volatility = []
    lev_stat = np.zeros(4)
    p_value = np.zeros(4)
    for i in range(0,4):
        election_volatility.append(election_data[election_data.columns[i]].std())
        non_election_volatility.append(non_election_data[election_data.columns[i]].std())

        # Annualize the volatility assuming 252 trading days per year
        annualized_election_volatility.append(election_volatility[i] * np.sqrt(252))
        annualized_non_election_volatility.append(non_election_volatility[i] * np.sqrt(252))

        # Levene's Test to see if the variances of Daily Returns are equal
        lev_stat[i], p_value[i] = stats.levene(election_data[election_data.columns[i]], non_election_data[election_data.columns[i]])

    return print("""For {} months before, and {} months after, the p-values are {}""".format(months_before, months_after, p_value))


for x in range(1,7):
    for y in range(1,7):
        vol_difference_test_two(x,y, r'C:\Users\rycba\projects\main\.vscode\MF703_Project\Data\TreasuryData\cleaned_treasury_data.csv')
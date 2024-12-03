# 
# Title: 
# Name: Aiden Perkins
# Email address: ajp15@bu.edu
# Description: Helper function to retrieve risk-free rate
# 
#
import pandas as pd

def get_risk_free_rate(dateString):
    df = pd.read_csv("../Data/TreasuryData/Cleaned-Data/cleaned_treasury_data.csv")
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    halfYear = df["1 Mo"].to_frame()
    rf = halfYear.loc[dateString, '1 Mo']
    rf = rf / 100
    return rf
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.stats import norm
from functools import reduce
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from DateRanges import electionPeriodBoolsDF
from FinancialInstrument import FinancialInstrument

class ETF(FinancialInstrument):
    """
    Class representing an Exchange-Traded Fund (ETF).
    """
    def __init__(self, ticker, period):
        """
        This is the constructor for the ETF class. Ticker is the string of the ticker and period is an integer
        that represents which time period is wanted to be used.
            

        Args:
            ticker (string): The ticker of the ETF.
            period (integer): The time period that should be considered.
                period = 1 -> election periods
                period = -1 -> non-election periods
                period = anything else -> total time period
        
        Returns:
            None
        """
        self.tickerCode = ticker
        self.df = pd.read_csv("../Data/ETFData/merged_cleaned_etf_data.csv")
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.prices = self.df.set_index('Date')[[ticker + " Adj Close"]]
        self.periodCode = period
        merged_df = pd.merge(self.prices, electionPeriodBoolsDF, left_index=True, right_index=True, how='inner')
        if period == 1:
            merged_df = merged_df[merged_df['In an Election Period']]
        elif period == -1:
            merged_df = merged_df[~merged_df['In an Election Period']]
        merged_df = merged_df.drop(columns=['In an Election Period'])
        self.prices = merged_df

    
    @property
    def log_returns(self):
        if hasattr(self, '_log_returns'):
            return self._log_returns
        return np.log(self.prices / self.prices.shift(1)).dropna().rename(columns={self.ticker + " Adj Close": self.ticker + " Log Return"})

    @log_returns.setter
    def log_returns(self, value):
        self._log_returns = value

    @property
    def instrument_type(self):
        return "ETF"
    
    @property
    def ticker(self):
        return self.tickerCode
    
    @property
    def period(self):
        if self.periodCode == 1:
            return "Election Periods"
        if self.periodCode == -1:
            return "Non-Election Periods"
        return "Total Time Period"


    def calculate_VaR(self, confidence_level=0.95):
        """
        This function calculates the Value at Risk (VaR) for log returns at a given confidence level.

        Args:
            confidence_level (float): The confidence level for the VaR calculation.
        
        Returns:
            VaR of the ETF.
        """
        mean = self.log_returns.mean()
        std_dev = self.log_returns.std()
        z_score = norm.ppf(1 - confidence_level)
        VaR = -(mean + z_score * std_dev)
        return float(VaR.iloc[0])
    
    def calculate_ES(self, confidence_level=0.95):
        """
        This function calculates the Expected Shortfall (ES) for the log returns at a given confidence level.

        Args:
            confidence_level (float): The confidence level for the ES calculation.
        
        Returns:
            ES of the asset.
        """
        VaR = self.calculate_VaR(confidence_level)
        tail_losses = self.log_returns[self.log_returns < VaR]
        ES = -tail_losses.mean()
        return float(ES.iloc[0])
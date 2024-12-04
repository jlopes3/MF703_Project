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

class Treasury(FinancialInstrument):
    """
    Class representing a Treasury.
    """
    def __init__(self, ticker):
        """
        This is the constructor for the Treasury class. Ticker is the string for the length.
            
        Args:
            ticker (string): The string for the length.
        
        Returns:
            None
        """
        self.tickerCode = ticker
        self.df = pd.read_csv("../Data/TreasuryData/Treasury-Price-Plus-Coupon-Data.csv")
        self.df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self._prices = self.df[ticker].to_frame()
        self.periodCode = 0
    
    @property
    def full_log_returns(self):
        return np.log(self._prices / self._prices.shift(1)).dropna().rename(columns={self.tickerCode: self.tickerCode + " Log Return"})

    @property
    def log_returns(self):
        if hasattr(self, '_log_returns'):
            return self._log_returns
        return np.log(self._prices / self._prices.shift(1)).dropna().rename(columns={self.tickerCode: self.tickerCode + " Log Return"})

    @log_returns.setter
    def log_returns(self, value):
        self._log_returns = value

    @property
    def period(self):
        if hasattr(self, '_period'):
            return self._period
        return self.periodCode

    @period.setter
    def period(self, value):
        self._period = value

    @property
    def instrument_type(self):
        return "Treasury"
    
    @property
    def ticker(self):
        return self.tickerCode

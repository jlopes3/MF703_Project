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

class Future(FinancialInstrument):
    """
    Class representing a Future.
    """
    def __init__(self, ticker):
        """
        This is the constructor for the Future class. Ticker is the string of the ticker.
            
        Args:
            ticker (string): The ticker of the Future.
        
        Returns:
            None
        """
        self.tickerCode = ticker
        self.df = pd.read_csv("../Data/FuturesData/merged_cleaned_futures_data.csv")
        self.df = self.df[self.df[ticker + " LAST"] >= 0]
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self._prices = self.df.set_index('Date')[[ticker + " LAST"]]
        self.periodCode = 0
    
    @property
    def full_log_returns(self):
        return np.log(self._prices / self._prices.shift(1)).dropna().rename(columns={self.tickerCode + " LAST": self.tickerCode + " Log Return"})

    @property
    def log_returns(self):
        if hasattr(self, '_log_returns'):
            return self._log_returns
        return np.log(self._prices / self._prices.shift(1)).dropna().rename(columns={self.tickerCode + " LAST": self.tickerCode + " Log Return"})

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
        return "Future"
    
    @property
    def ticker(self):
        return self.tickerCode


    # These functions are not used anywhere so I am commenting them out for now
    # def calculate_VaR(self, confidence_level=0.95):
    #     """
    #     This function calculates the Value at Risk (VaR) for log returns at a given confidence level.

    #     Args:
    #         confidence_level (float): The confidence level for the VaR calculation.
        
    #     Returns:
    #         Series with VaR of each asset.
    #     """
    #     mean = self.log_returns.mean()
    #     std_dev = self.log_returns.std()
    #     z_score = norm.ppf(1 - confidence_level)
    #     VaR = -(mean + z_score * std_dev)
    #     return float(VaR.iloc[0])
    
    # def calculate_ES(self, confidence_level=0.95):
    #     """
    #     This function calculates the Expected Shortfall (ES) for the log returns at a given confidence level.

    #     Args:
    #         confidence_level (float): The confidence level for the ES calculation.
        
    #     Returns:
    #         Series with ES of each asset.
    #     """
    #     VaR = self.calculate_VaR(confidence_level)
    #     tail_losses = self.log_returns[self.log_returns < VaR]
    #     ES = -tail_losses.mean()
    #     return float(ES.iloc[0])
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.stats import norm
from functools import reduce
from datetime import date
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from DateRanges import electionPeriodBoolsDF

class FinancialInstrument(ABC):
    """
    Abstract base class representing a generic financial instrument.
    """

    @property
    @abstractmethod
    def full_log_returns(self):
        """
        Abstract property representing the full log returns as a pandas DataFrame.
        This is the log returns for all available data, and this is not updated by
        filtering.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def log_returns(self):
        """
        Abstract property representing the log returns as a pandas DataFrame. This
        is updated by filtering.
        Must be implemented by subclasses.
        """
        pass

    @log_returns.setter
    def log_returns(self, value):
        pass

    @property
    @abstractmethod
    def period(self):
        """
        Abstract property representing the period as a an integer. 1 correspends to
        an election period, -1 corresponds to a non-election period, and anything
        else corresponds to the total time period.
        Must be implemented by subclasses.
        """
        pass

    @period.setter
    def period(self, value):
        pass
    
    @property
    @abstractmethod
    def instrument_type(self):
        """
        Abstract property representing the instrument type as a string.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def ticker(self):
        """
        Abstract property representing the ticker as a string.
        Must be implemented by subclasses.
        """
        pass
    
    def expected_annualized_log_return(self):
        return float((self.log_returns.mean()*252).iloc[0])
    
    def get_date_range(self):
        return self.log_returns.index.min(), self.log_returns.index.max()

    def filter(self, startDate=date(1800, 1, 1), endDate=date(2100, 12, 31), period=0):
        self.period = period
        filtered = self.full_log_returns.loc[startDate:endDate]
        merged_df = pd.merge(filtered, electionPeriodBoolsDF, left_index=True, right_index=True, how='inner')
        if period == 1:
            merged_df = merged_df[merged_df['In an Election Period']]
        elif period == -1:
            merged_df = merged_df[~merged_df['In an Election Period']]
        merged_df = merged_df.drop(columns=['In an Election Period'])
        self.log_returns = merged_df
        
    def calculate_volatility(self, annualize=True):
        """
        This function calculates the daily or annualized volatility of log returns.

        Args:
            annualize (Boolean): If True, returns annualized volatility. Else, returns daily volatility.
        
        Returns:
            Series with volatility of each asset.
        """
        daily_vol = self.log_returns.std()
        if annualize:
            return float(daily_vol.iloc[0]) * np.sqrt(252)
        return float(daily_vol.iloc[0])
    
    def calculate_variance(self, annualize=True):
        """
        This function calculates the daily or annualized variance of log returns.

        Args:
            annualize (Boolean): If True, returns annualized variance. Else, returns daily variance.
        
        Returns:
            Float with the variance of the log returns.
        """
        daily_variance = self.log_returns.var()
        if annualize:
            return float(daily_variance.iloc[0]) * 252
        return float(daily_variance.iloc[0])
    
    def correlation_matrix(self, others):
        """
        This function calculates the correlation matrix of the log returns.

        Args:
            otherList (list): List of other Financial Instruments.

        Returns:
            DataFrame with correlation matrix.
        """
        log_return_df_list = [self.log_returns]
        for other in others:
            log_return_df_list += [other.log_returns]
        merged_df = reduce(lambda x, y: pd.merge(x, y, how='inner', left_index=True, right_index=True), log_return_df_list)
        return merged_df.corr()
    
    def covariance_matrix(self, others):
        """
        This function calculates the covariance matrix of the log returns.

        Args:
            otherList (list): List of other Financial Instruments.

        Returns:
            DataFrame with covariance matrix.
        """
        log_return_df_list = [self.log_returns]
        for other in others:
            log_return_df_list += [other.log_returns]
        merged_df = reduce(lambda x, y: pd.merge(x, y, how='inner', left_index=True, right_index=True), log_return_df_list)
        return merged_df.cov()*252

    def total_log_return(self):
        """
        This function calculates the total log returns.

        Returns:
            Float
        """
        return float(self.log_returns.sum().iloc[0])
    
    def calculate_beta(self, benchmark):
        """
        This function calculates the beta between self and a benchmark Financial Instrument.

        Args:
            benchmark (FinancialInstrument): benchmark Financial Instrument.

        Returns:
            Float
        """
        cov_matrix = self.covariance_matrix([benchmark])
        covariance = cov_matrix.iloc[0, 1]
        benchmark_var = benchmark.calculate_variance(annualize=False) # There was an issue with dividing by zero when working with ETFs, check on this again
        benchmark_var = cov_matrix.iloc[1,1]
        beta = float(covariance / benchmark_var)
        return beta
    
    def summary(self):
        period_string = ""
        if self.period == 1:
            period_string = "Election Periods Only"
        elif self.period == -1:
            period_string = "Non-Election Periods Only"
        else:
            period_string = "All Periods"
        vol = self.calculate_volatility()
        first_date = self.log_returns.index[0]
        last_date = self.log_returns.index[-1]
        return (f"Type: {self.instrument_type}, "
                f"Ticker: {self.ticker}, "
                f"Period: {period_string}, "
                f"Volatility: {vol}, "
                f"First Date: {first_date}, "
                f"Last Date: {last_date}")
    
    def __str__(self):
        return self.summary()


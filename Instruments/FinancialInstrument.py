from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.stats import norm
from functools import reduce

class FinancialInstrument(ABC):
    """
    Abstract base class representing a generic financial instrument.
    """

    @property
    @abstractmethod
    def log_returns(self):
        """
        Abstract property representing the log returns as a pandas DataFrame.
        Must be implemented by subclasses.
        """
        pass

    @log_returns.setter
    def log_returns(self, value):
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

    @property
    @abstractmethod
    def period(self):
        """
        Abstract property representing the type of period as a string.
        Must be implemented by subclasses.
        """
        pass

    def filter(self, startDate, endDate):
        filtered = self.log_returns.loc[startDate:endDate]
        self.log_returns = filtered
        
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
        return merged_df.cov()

    def calculate_beta(self, benchmark):
        """
        This function calculates the beta between self and a benchmark Financial Instrument.

        Args:
            benchmark (FinancialInstrument): benchmark Financial Instrument.

        Returns:
            DataFrame with correlation matrix.
        """
        cov_matrix = self.covariance_matrix([benchmark])
        covariance = cov_matrix.iloc[0, 1]
        benchmark_var = benchmark.calculate_variance(annualize=False) # There was an issue with dividing by zero when working with ETFs, check on this again
        benchmark_var = cov_matrix.iloc[1,1]
        beta = float(covariance / benchmark_var)
        return beta
    
    def summary(self):
        vol_series = self.calculate_volatility()
        vol = float(vol_series.iloc[0])
        return (f"Type: {self.instrument_type}, "
                f"Ticker: {self.ticker}, "
                f"Period: {self.period}, "
                f"Volatility: {vol}")
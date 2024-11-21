import numpy as np
import pandas as pd
from FinancialInstrument import FinancialInstrument
from ETF import ETF
from Future import Future
from datetime import date
from DateRanges import electionPeriodBoolsDF, e_year_ranges

class Portfolio:
    """
    Class representing a portfolio of financial instruments.
    """

    def __init__(self, instrument_weight_list):
        """
        Initialize a portfolio.

        Args:
            instrument_weight_list (FinancialInstrument, float): List of tuples where the the first element
                is the FinancialInstrument and the second element is the weight as a float.
        """
        self.instruments, self.weights = zip(*instrument_weight_list)
        self.weights = np.array(self.weights)
        self._validate_weights()
        self.start_date = (max(self.instruments, key=lambda instrument: instrument.get_date_range()[0])).get_date_range()[0]
        self.end_date = (min(self.instruments, key=lambda instrument: instrument.get_date_range()[1])).get_date_range()[1]
        for instrument in self.instruments:
            instrument.filter(self.start_date, self.end_date)
        log_returns_dict = {
            instrument.ticker: instrument.log_returns for instrument in self.instruments
        }
        self.full_asset_log_returns_df = pd.concat(log_returns_dict, axis=1)
        self.asset_log_returns_df = self.full_asset_log_returns_df
        self.portfolio_log_returns = self.asset_log_returns_df.dot(self.weights)

    def filter(self, startDate=date(1800, 1, 1), endDate=date(2100, 12, 31), period=0):
        for instrument in self.instruments:
            instrument.filter(startDate=startDate, endDate=endDate, period=period)
        log_returns_dict = {
            instrument.ticker: instrument.log_returns for instrument in self.instruments
        }
        self.asset_log_returns_df = pd.concat(log_returns_dict, axis=1)
        self.portfolio_log_returns = self.asset_log_returns_df.dot(self.weights)
        
    def _validate_weights(self):
        """
        Validate that the sum of weights equals 1.
        """
        total_weight = sum(self.weights)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Portfolio weights must sum to 1. Current sum: {total_weight}")
    
    def change_weights(self, new_weights):
        self._validate_weights()
        self.weights = np.array(new_weights)
        self.portfolio_log_returns = self.asset_log_returns_df.dot(self.weights)
        return

    def total_log_return(self):
        """
        Calculate the expected log return of the portfolio.

        Returns:
            float: The expected return of the portfolio.
        """
        total_log_return = 0
        for instrument, weight in zip(self.instruments, self.weights):
            total_log_return += weight * instrument.total_log_return()
        return total_log_return

    def covariance_matrix(self):
        """
        Calculate the covariance matrix for the portfolio's instruments.

        Returns:
            np.ndarray: Covariance matrix of the portfolio's instruments.
        """
        return self.instruments[0].covariance_matrix(self.instruments[1:])
    
    def portfolio_variance(self):
        """
        Calculate the portfolio's variance.

        Returns:
            float: The portfolio variance.
        """
        return np.dot(self.weights.T, np.dot(self.covariance_matrix(), self.weights))
    
    def portfolio_volatility(self):
        """
        Calculate the portfolio's volatility.

        Returns:
            float: The portfolio volatility.
        """
        return np.sqrt(self.portfolio_variance())
    
    def calculate_beta(self, benchmark):
        total_beta = 0
        for instrument, weight in zip(self.instruments, self.weights):
            total_beta += weight * instrument.calculate_beta(benchmark)
        return total_beta
        

    def summary(self):
        """
        Print a summary of the portfolio.

        Returns:
            str: Summary of the portfolio.
        """
        summary_lines = [f"Instrument: {inst.ticker}, Weight: {weight}" 
                         for inst, weight in zip(self.instruments, self.weights)]
        return "\n".join(summary_lines) + f"\nExpected Log Return: {self.total_log_return():.4}\n" \
                                          f"Portfolio Volatility: {self.portfolio_volatility():.2}"
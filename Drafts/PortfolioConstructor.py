import pandas as pd
import numpy as np
import datetime as date
from scipy.optimize import minimum
from FinancialInstrument import FinancialInstrument
from ETF import ETF
from Future import Future
from DateRanges import electionPeriodBoolsDF, e_year_ranges
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from ExpectedReturnCalc import ExpectedReturnsCalculator

class PortfolioConstructor:
    def __init__(self, instrument_weight_list):
        """
        Initialize a portfolio.

        Args:
            instrument_weight_list (FinancialInstrument, float): List of tuples where the the first element
                is the FinancialInstrument and the second element is the weight as a float.
        """
        self.period = "Total time period"
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
        if period == 0:
            self.period = "Total time period"
        elif period == 1:
            self.period = "Election period"
        elif period == -1:
            self.period = "Non-election period"
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
        
    def portfolio_performance(self, weights=None):
        """
        Calculate portfolio performance metrics: return, risk, and Sharpe ratio.
        """
        if weights is None:
            weights = self.weights
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        return portfolio_return, portfolio_risk, sharpe_ratio

    def optimize_weights(self, target_return=None):
        """
        Optimize portfolio weights for maximum Sharpe ratio or minimum risk for a target return.
        """
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix, self.risk_free_rate)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return:
            constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - target_return})
        
        bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
        initial_weights = np.ones(num_assets) / num_assets  # Equal weights initially

        if target_return:
            result = minimize(self._portfolio_risk, initial_weights, args=args, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        else:
            result = minimize(self._negative_sharpe_ratio, initial_weights, args=args, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise ValueError("Optimization failed:", result.message)
        
        self.weights = result.x  # Update portfolio weights
        return self.weights

    def _portfolio_risk(self, weights, *args):
        """
        Calculate portfolio risk (standard deviation).
        """
        _, cov_matrix, _ = args
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def _negative_sharpe_ratio(self, weights, *args):
        """
        Calculate negative Sharpe ratio for optimization.
        """
        mean_returns, cov_matrix, risk_free_rate = args
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_risk

    def summary(self):
        """
        Generate a summary of the portfolio performance.
        """
        portfolio_return, portfolio_risk, sharpe_ratio = self.portfolio_performance()
        return {
            "Weights": self.weights,
            "Expected Return": portfolio_return,
            "Risk (Std Dev)": portfolio_risk,
            "Sharpe Ratio": sharpe_ratio,
        }

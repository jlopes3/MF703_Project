import numpy as np
import pandas as pd
from FinancialInstrument import FinancialInstrument
from ETF import ETF
from Future import Future
from datetime import date
from DateRanges import electionPeriodBoolsDF, e_year_ranges
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from ExpectedReturnCalc import ExpectedReturnsCalculator
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as sc
from scipy.stats import norm
from scipy.optimize import minimize

class Portfolio:
    """
    Class representing a portfolio of financial instruments.
    """

    def __init__(self, instrument_list):
        """
        Initialize a portfolio.

        Args:
            instrument_weight_list (FinancialInstrument, float): List of tuples where the the first element
                is the FinancialInstrument and the second element is the weight as a float.
        """
        self.period = "Total time period"
        self.instruments = instrument_list
        self.even_weights = np.array([1./len(instrument_list)]*len(instrument_list))
        self.weights = self.even_weights
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
        expected_annualized_log_returns_dict = {
            instrument.ticker + " Expected Annualized Log Return": instrument.expected_annualized_log_return() for instrument in self.instruments
        }
        self.expected_annaulized_log_returns_df = pd.Series(expected_annualized_log_returns_dict)

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
        expected_annualized_log_returns_dict = {
            instrument.ticker + " Expected Annualized Log Return": instrument.expected_annualized_log_return() for instrument in self.instruments
        }
        self.expected_annaulized_log_returns_df = pd.Series(expected_annualized_log_returns_dict)
        
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

    def annualized_covariance_matrix(self):
        """
        Calculate the annualized covariance matrix for the portfolio's instruments.

        Returns:
            np.ndarray: Covariance matrix of the portfolio's instruments.
        """
        return self.asset_log_returns_df.cov() * 252
    
    def annualized_portfolio_vol(self, weights, cov_mat):
        return (weights.T @ cov_mat @ weights)**0.5
    
    def expected_portfolio_return(self, weights, expected_returns):  
        return weights.T @ expected_returns
    
    def minimize_vol(self, target_return, expected_return, cov_mat):
        n = expected_return.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((-0.5, 0.5),)*n
        return_is_target = {
            'type' : 'eq',
            'args' : (expected_return,),
            'fun' : lambda weights, expected_return : target_return - self.expected_portfolio_return(weights, expected_return)
        }
        weights_sum_to_1 = {
            'type' : 'eq',
            'fun' : lambda weights : np.sum(weights) - 1
        }
        results = minimize(self.annualized_portfolio_vol, init_guess,
                        args = (cov_mat,), method = 'SLSQP',
                        options={'disp': False},
                        constraints= (return_is_target, weights_sum_to_1),
                        bounds = bounds
                        )
        return results.x
    
    def optimal_weights(self, n_points, expected_return, cov_mat):
        target_rs = np.linspace(expected_return.min(), expected_return.max(), n_points)
        weights = [self.minimize_vol(target_return, expected_return, cov_mat) for target_return in target_rs]
        return weights
    
    def plot_ef(self, n_points=1000):
        cov = self.annualized_covariance_matrix()
        weights = self.optimal_weights(n_points, self.expected_annaulized_log_returns_df, cov)
        rets = [self.expected_portfolio_return(w, self.expected_annaulized_log_returns_df) for w in weights]
        vols = np.array([self.annualized_portfolio_vol(w, cov) for w in weights])
        ef = pd.DataFrame({'Returns': rets, 'Volatility': vols})
        fig = ef.plot(x='Volatility', y='Returns', style='.-', color='green')
        return fig

    def summary(self):
        """
        Print a summary of the portfolio.

        Returns:
            str: Summary of the portfolio.
        """
        start_date = self.portfolio_log_returns.index[0]
        end_date = self.portfolio_log_returns.index[-1]

        return f"Period: {self.period}\n" + \
            "\n".join([f"Instrument: {inst.ticker}, Weight: {weight}" 
                        for inst, weight in zip(self.instruments, self.weights)]) + \
            f"\nExpected Log Return: {self.total_log_return():.4}\n" \
            f"Portfolio Volatility: {self.portfolio_volatility():.2}\n" \
            f"Start Date: {start_date}\nEnd Date: {end_date}"
    
    def __str__(self):
        return self.summary()





    
    def calculate_beta(self, benchmark):
        total_beta = 0
        for instrument, weight in zip(self.instruments, self.weights):
            total_beta += weight * instrument.calculate_beta(benchmark)
        return total_beta
    
    def calculate_expected_return(self,benchmark,rf):
        """
        Uses CAPM on the portfolio to determine the expected return
        
        Args:
            benchmark (financial instrument): Financial Instrument class representing a benchmark
            rf (double): risk-free rate as an annualized figure

        Returns:
            float: portfolio expected return
        """
        beta = self.calculate_beta(benchmark)
        rm = benchmark.full_log_returns
        rm = rm.mean()
        rm *= 252
        risk_premium = rm - rf
        expected_return = beta * risk_premium + rm
        return expected_return
    
    def sharpe_ratio(self,benchmark,rf):
        """
        Calculates and returns the sharpe ratio of the portfolio
        
        Args:
            benchmark (financial instrument): Financial Instrument class representing benchmark comparison
            rf (double): risk-free rate annualized as log
        
        Returns:
            float: portfolio Sharpe Ratio
        """
        exp_return = self.calculate_expected_return(benchmark, rf)
        vol = self.portfolio_volatility()
        return exp_return / vol
        
            
    def get_start_date(self):
        """
        return the start date of the portfolio

        Returns:
            start date of portfolio

        """
        return self.portfolio_log_returns.index[0]
    def get_end_date(self):
        """
        return the end date of the portfolio

        Returns:
            end date of portfolio

        """
        return self.portfolio_log_returns.index[-1]
    
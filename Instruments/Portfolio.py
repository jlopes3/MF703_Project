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
import yfinance as yf

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
        
        # Gets all instruments to have the same time period
        self.start_date = (max(self.instruments, key=lambda instrument: instrument.get_date_range()[0])).get_date_range()[0]
        self.end_date = (min(self.instruments, key=lambda instrument: instrument.get_date_range()[1])).get_date_range()[1]
        for instrument in self.instruments:
            instrument.filter(self.start_date, self.end_date)
        
        # Combines the log returns of each instrument into a dataframe
        log_returns_dict = {
            instrument.ticker: instrument.log_returns for instrument in self.instruments
        }
        self.full_asset_log_returns_df = pd.concat(log_returns_dict, axis=1) # Should not change with filtering
        self.asset_log_returns_df = self.full_asset_log_returns_df
        
        # Gets the expected annualized log returns for each instrument into a Series
        # The logic of this may need to change based on how we want to calculate expected
        # return values for each instrument/portfolio
        expected_annualized_log_returns_dict = {
            instrument.ticker + " Expected Annualized Log Return": instrument.expected_annualized_log_return() for instrument in self.instruments
        }
        self.expected_annualized_log_returns_df = pd.Series(expected_annualized_log_returns_dict)

    def filter(self, startDate=date(1800, 1, 1), endDate=date(2100, 12, 31), period=0):
        if period == 0:
            self.period = "Total time period"
        elif period == 1:
            self.period = "Election period"
        elif period == -1:
            self.period = "Non-election period"
        
        # Resets self.asset_log_returns_df using filtered data from each instrument
        for instrument in self.instruments:
            instrument.filter(startDate=startDate, endDate=endDate, period=period)
        log_returns_dict = {
            instrument.ticker: instrument.log_returns for instrument in self.instruments
        }
        self.asset_log_returns_df = pd.concat(log_returns_dict, axis=1)

        # Gets the expected annualized log returns for each instrument into a Series
        # The logic of this may need to change based on how we want to calculate expected
        # return values for each instrument/portfolio
        expected_annualized_log_returns_dict = {
            instrument.ticker + " Expected Annualized Log Return": instrument.expected_annualized_log_return() for instrument in self.instruments
        }
        self.expected_annualized_log_returns_df = pd.Series(expected_annualized_log_returns_dict)

    
    def annualized_covariance_matrix(self):
        """
        Calculate the annualized covariance matrix for the portfolio's instruments.

        Returns:
            np.ndarray: Covariance matrix of the portfolio's instruments.
        """
        return self.asset_log_returns_df.cov() * 252
    
    def annualized_portfolio_vol(self, weights):
        return (weights.T @ self.annualized_covariance_matrix() @ weights)**0.5
    
    # This function will likely need to be redone. This calculates the portfolio's
    # expected annualized log return by simply summing the weighted expected annualized
    # log return of each instrument
    def expected_portfolio_return(self, weights):  
        return weights.T @ self.expected_annualized_log_returns_df
    
    # This function minimizes volatility for a given portfolio target return. It relies on
    # portfolio.expected_portfolio_return(weights), so that function needs to be looked at
    # more carefully. The bounds are set to -0.5 and 0.5 for the weights of any individual
    # instrument.
    def minimize_vol(self, target_return):
        n = self.expected_annualized_log_returns_df.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((-0.5, 0.5),)*n  # We can adjust the constraints for individual asset weights here
        return_is_target = {
            'type' : 'eq',
            'args' : (self.expected_annualized_log_returns_df,),
            'fun' : lambda weights, expected_return : target_return - self.expected_portfolio_return(weights)
        }
        weights_sum_to_1 = {
            'type' : 'eq',
            'fun' : lambda weights : np.sum(weights) - 1
        }
        results = minimize(self.annualized_portfolio_vol, init_guess,
                        method = 'SLSQP',
                        options={'disp': False},
                        constraints= (return_is_target, weights_sum_to_1),
                        bounds = bounds
                        )
        return results.x
    
    # This function gets a list of list of weights for the minimal volatility portfolio for
    # various target returns. The various target returns are decided by taking n_points
    # target returns between the lowest indiviudal expected_annualized_log_return of the
    # instruments and the highest indiviudal expected_annualized_log_return of the instruments.
    def optimal_weights(self, n_points):
        target_rs = np.linspace(self.expected_annualized_log_returns_df.min(), self.expected_annualized_log_returns_df.max(), n_points)
        weights = [self.minimize_vol(target_return) for target_return in target_rs]
        return weights
    
    # This function plots the efficient frontier by plotting the points by getting the optimal
    # weights for n_points target returns. It uses the optimal weights to calculate the annualized
    # portfolio volatility for each target return, and plots the returns as a function of the
    # volatility.
    def plot_ef(self, n_points=1000):
        weights = self.optimal_weights(n_points)
        rets = [self.expected_portfolio_return(w) for w in weights]
        vols = np.array([self.annualized_portfolio_vol(w) for w in weights])
        ef = pd.DataFrame({'Returns': rets, 'Volatility': vols})
        ef.plot(x='Volatility', y='Returns', style='.-', color='green')
        return
    
    # This function calculates the maximum Sharpe ratio portfolio for a given period. It
    # calculates n_points optimal weights and then calculates the volatility and returns
    # for each of the optimal weights. It then calculates the sharpe ratio for each of the
    # optimal weights. It then selects the best sharpe ratio and returns the maximum sharpe
    # ratio and the corresponding returns, volatility, and weights. date is the day to get
    # the risk free rate from. The risk free rate is from 10 Year Treasury yield.
    def max_sharpe_portfolio(self, date, n_points=1000):
        ticker = yf.Ticker("^TNX")
        print(ticker)
        data = ticker.history(start=date, end=date)
        risk_free_rate = data["Close"].iloc[-1] / 100
        log_risk_free_rate = np.log(1 + risk_free_rate)
        weights = self.optimal_weights(n_points)
        rets = [self.expected_portfolio_return(w) for w in weights]
        sharpe_list = []
        for ret, w in zip(rets, weights):
            sharpe_ratio = ((ret - log_risk_free_rate) / self.annualized_portfolio_vol(w))
            sharpe_list += [(sharpe_ratio, ret, w)]
        max_sharpe, max_sharpe_ret, max_sharpe_weights = max(sharpe_list, key=lambda x: x[0])
        max_sharp_vol = self.annualized_portfolio_vol(max_sharpe_weights)
        return max_sharpe, max_sharpe_ret, max_sharp_vol, max_sharpe_weights



    def summary(self):
        """
        Print a summary of the portfolio.

        Returns:
            str: Summary of the portfolio.
        """
        start_date = self.portfolio_log_returns.index[0]
        end_date = self.portfolio_log_returns.index[-1]

        return f"Period: {self.period}\n" + \
            "\n".join([f"Instrument: {inst.ticker}" 
                        for inst in self.instruments]) + \
            f"Start Date: {start_date}\nEnd Date: {end_date}"
    
    def __str__(self):
        return self.summary()

    # Commented these out for now because they are not used yet

    # def calculate_beta(self, benchmark, weights):
    #     total_beta = 0
    #     for instrument, weight in zip(self.instruments, weights):
    #         total_beta += weight * instrument.calculate_beta(benchmark)
    #     return total_beta
    
    # def calculate_expected_return(self,benchmark,rf):
    #     """
    #     Uses CAPM on the portfolio to determine the expected return
        
    #     Args:
    #         benchmark (financial instrument): Financial Instrument class representing a benchmark
    #         rf (double): risk-free rate as an annualized figure

    #     Returns:
    #         float: portfolio expected return
    #     """
    #     beta = self.calculate_beta(benchmark)
    #     rm = benchmark.full_log_returns
    #     rm = rm.mean()
    #     rm *= 252
    #     risk_premium = rm - rf
    #     expected_return = beta * risk_premium + rm
    #     return expected_return
    
    # def sharpe_ratio(self,benchmark,rf):
    #     """
    #     Calculates and returns the sharpe ratio of the portfolio
        
    #     Args:
    #         benchmark (financial instrument): Financial Instrument class representing benchmark comparison
    #         rf (double): risk-free rate annualized as log
        
    #     Returns:
    #         float: portfolio Sharpe Ratio
    #     """
    #     exp_return = self.calculate_expected_return(benchmark, rf)
    #     vol = self.portfolio_volatility()
    #     return exp_return / vol
             
    # def get_start_date(self):
    #     """
    #     return the start date of the portfolio

    #     Returns:
    #         start date of portfolio

    #     """
    #     return self.asset_log_returns_df.index[0]
    # def get_end_date(self):
    #     """
    #     return the end date of the portfolio

    #     Returns:
    #         end date of portfolio

    #     """
    #     return self.asset_log_returns_df.index[-1]
    
import numpy as np
import pandas as pd
from FinancialInstrument import FinancialInstrument
from ETF import ETF
from Future import Future
from datetime import date
from Instruments.DateRanges import electionPeriodBoolsDF, e_year_ranges
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
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

    def __init__(self, instrument_list, rf=0, equity_benchmark=None, future_benchmark=None, treasury_benchmark=None):
        """
        Initialize a portfolio.

        Args:
            instrument_weight_list (FinancialInstrument, float): List of tuples where the the first element
                is the FinancialInstrument and the second element is the weight as a float.
            rf (float): Risk-free rate to be used in CAPM formula and Sharpe Ratio. Not logarithm
            equity_benchmark (FinancialInstrument): Financial Instrument for the market in CAPM formula
            future_benchmark (FinancialInstrument): Financial Instrument for the future in CAPM formula
            treasury_benchmark (FinancialInstrument): Financial Instrument for the treasury in CAPM formula
        """
        self.period = "Total time period"
        self.instruments = instrument_list
        self.rf = rf
        self.equity_benchmark = equity_benchmark
        self.future_benchmark = future_benchmark
        self.treasury_benchmark = treasury_benchmark
        
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
        self.full_asset_log_returns_df = self.full_asset_log_returns_df.dropna()
        self.asset_log_returns_df = self.full_asset_log_returns_df.dropna()


        
        # Gets the expected annualized log returns for each instrument into a Series
        # The logic of this may need to change based on how we want to calculate expected
        # return values for each instrument/portfolio
        expected_annualized_log_returns_dict = {}
        for instrument in self.instruments:
            if instrument.instrument_type == "ETF":
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.equity_benchmark)
            elif instrument.instrument_type == "Future":
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.future_benchmark)
            else:
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.treasury_benchmark)    
        self.expected_annualized_log_returns_df = pd.Series(expected_annualized_log_returns_dict)

    def set_rf(self, new_rf):
        self.rf = new_rf
        expected_annualized_log_returns_dict = {}
        for instrument in self.instruments:
            if instrument.instrument_type == "ETF":
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.equity_benchmark)
            elif instrument.instrument_type == "Future":
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.future_benchmark)
            else:
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.treasury_benchmark)    
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
        expected_annualized_log_returns_dict = {}
        for instrument in self.instruments:
            if instrument.instrument_type == "ETF":
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.equity_benchmark)
            elif instrument.instrument_type == "Future":
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.future_benchmark)
            else:
                expected_annualized_log_returns_dict[instrument.ticker + " Expected Annualized Log Return"] = instrument.expected_annualized_log_return(self.rf, self.treasury_benchmark)    
        self.expected_annualized_log_returns_df = pd.Series(expected_annualized_log_returns_dict)

    def historical_annualized_log_return(self, weights):
        # Gets the historical returns of the portfolio using the given weights
        # and the time period decided by the filter function
        # Annualizes the log returns of each asset and then dots them with the
        # weights
        return ((self.asset_log_returns_df.mean() * 252) * weights).sum()

    
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
    def minimize_vol(self, target_return,min_weight=-0.5,max_weight=0.5):
        n = self.expected_annualized_log_returns_df.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((min_weight, max_weight),)*n  # We can adjust the constraints for individual asset weights here
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
    def optimal_weights(self, n_points, min_weight=-.5, max_weight=.5):
        target_rs = np.linspace(self.expected_annualized_log_returns_df.min(), self.expected_annualized_log_returns_df.max(), n_points)
        weights = [self.minimize_vol(target_return,min_weight,max_weight) for target_return in target_rs]
        return weights
    
    # This function plots the efficient frontier by plotting the points by getting the optimal
    # weights for n_points target returns. It uses the optimal weights to calculate the annualized
    # portfolio volatility for each target return, and plots the returns as a function of the
    # volatility.
    def plot_ef(self, n_points=1000, min_weight=-.5, max_weight=.5):
        weights = self.optimal_weights(n_points,min_weight,max_weight)
        rets = [self.expected_portfolio_return(w) for w in weights]
        vols = np.array([self.annualized_portfolio_vol(w) for w in weights])
        sharpe_list = []
        for ret, vol, w in zip(rets, vols, weights):
            sharpe_ratio = ((ret - np.log(1 + self.rf)) / vol)
            sharpe_list += [(sharpe_ratio, ret, w, vol)]
        max_sharpe, max_sharpe_ret, max_sharpe_weights, max_sharpe_vol = max(sharpe_list, key=lambda x: x[0])
        ef = pd.DataFrame({'Annualized Log Return': rets, 'Volatility': vols})
        plain_graph = ef.plot(x='Volatility', y='Annualized Log Return', style='.-', color='green', markersize=0, legend=False, title="Efficient Frontier")
        plain_graph.set_ylabel('Annualized Log Return')
        graph = ef.plot(x='Volatility', y='Annualized Log Return', style='.-', color='green', markersize=0, legend=False, label="Efficient Frontier")
        vol_coords = [0, max_sharpe_vol, max_sharpe_vol + .04]
        ret_coords = [np.log(1 + self.rf), max_sharpe_ret, ((((max_sharpe_ret - np.log(1 + self.rf)) / max_sharpe_vol) * 0.04) + max_sharpe_ret)]
        graph.plot(vol_coords, ret_coords, color='red', linestyle='solid', label="Capital Market Line")
        graph.set_ylabel('Annualized Log Return')
        graph.scatter([max_sharpe_vol], [max_sharpe_ret], color='blue', s=20, label="Maximum Sharpe Portfolio")
        graph.legend(loc='lower right')
        return max_sharpe, max_sharpe_ret,  max_sharpe_vol, max_sharpe_weights
    
    # This function calculates the maximum Sharpe ratio portfolio for a given period. It
    # calculates n_points optimal weights and then calculates the volatility and returns
    # for each of the optimal weights. It then calculates the sharpe ratio for each of the
    # optimal weights. It then selects the best sharpe ratio and returns the maximum sharpe
    # ratio and the corresponding returns, volatility, and weights. date is the day to get
    # the risk free rate from. The risk free rate is from 10 Year Treasury yield.
    def max_sharpe_portfolio(self, n_points=1000, min_weight=-.5, max_weight=.5):
        log_risk_free_rate = np.log(1 + self.rf)
        weights = self.optimal_weights(n_points, min_weight, max_weight)
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
        start_date = self.asset_log_returns_df.index[0]
        end_date = self.asset_log_returns_df.index[-1]

        return f"\nPeriod: {self.period}\n" + \
            f"Risk-free rate: {self.rf}\n" + \
            "\n".join([f"Instrument: {inst.ticker}" 
                        for inst in self.instruments]) + \
            f"\nStart Date: {start_date}\nEnd Date: {end_date}\n"
    
    def portfolio_VaR(self,weights,conf_level = .95):
        """
        Calculate the Value at Risk of the current portfolio
        
        Args:
            conf_level (float): confidence level of calculation
            weights: vector of weights of items in portfolio
        Returns:
            float: VaR 
        """
        z_score = norm.ppf(conf_level)
        var = z_score * self.annualized_portfolio_vol(weights)
        return var
    
    def portfolio_ES(self,weights,conf_level=.95):
        """
        Calculate the Expected Shortfall of the current portfolio

        Args:
            conf_level (float): confidence level of calculation
            weights: vector of weights of items in portfolio
        Returns:
            float: ES
        """
        z_score = norm.ppf(conf_level)
        phi = norm.pdf(z_score)
        es = self.annualized_portfolio_vol(weights)*(phi/(1-conf_level))
        return es

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
    
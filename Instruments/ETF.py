import numpy as np
import pandas as pd
from scipy.stats import norm
from functools import reduce

class ETF:
    def __init__(self, ticker):
        """
        This function initializes the class containing ETF data give a ticker.

        Args:
            ticker (string): String for the ticker of the ETF.
        
        Returns:
            None    
        """
        self.df = pd.read_csv("../Data/ETFData/merged_cleaned_etf_data.csv")
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.prices = self.df.set_index('Date')[[ticker + " Adj Close"]]
        self.log_returns = self._calculate_log_returns().rename(columns={ticker + " Adj Close": ticker + " Log Return"})
   
    def _calculate_log_returns(self):
        """
        This function calculates the daily log returns from price data.

        Returns:
            DataFrame of daily log returns.
        """
        return np.log(self.prices / self.prices.shift(1)).dropna()
   
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
            return daily_vol * np.sqrt(252)
        return daily_vol
   
    def calculate_VaR(self, confidence_level=0.95):
        """
        This function calculates the Value at Risk (VaR) for log returns at a given confidence level.

        Args:
            confidence_level (float): The confidence level for the VaR calculation.
        
        Returns:
            Series with VaR of each asset.
        """
        mean = self.log_returns.mean()
        std_dev = self.log_returns.std()
        z_score = norm.ppf(1 - confidence_level)
        return -(mean + z_score * std_dev)
   
    def calculate_ES(self, confidence_level=0.95):
        """
        This function calculates the Expected Shortfall (ES) for the log returns at a given confidence level.

        Args:
            confidence_level (float): The confidence level for the ES calculation.
        
        Returns:
            Series with ES of each asset.
        """
        VaR = self.calculate_VaR(confidence_level)
        tail_losses = self.log_returns[self.log_returns < -VaR]
        return -tail_losses.mean()

    def correlation_matrix(self, otherList):
        """
        This function calculates the correlation matrix of the log returns.

        Args:
            otherList (list): List of strings for tickers for other ETFs.

        Returns:
            DataFrame with correlation matrix.
        """
        log_return_df_list = [self.log_returns]
        for ticker in otherList:
            log_return_df_list += [ETF(ticker).log_returns]
        merged_df = reduce(lambda x, y: pd.merge(x, y, how='inner', on="Date"), log_return_df_list)
        return merged_df.corr()
    
    def covariance_matrix(self, otherList):
        """
        This function calculates the covariance matrix of the log returns.

        Args:
            otherList (list): List of strings for tickers for other ETFs.

        Returns:
            DataFrame with correlation matrix.
        """
        log_return_df_list = [self.log_returns]
        for ticker in otherList:
            log_return_df_list += [ETF(ticker).log_returns]
        merged_df = reduce(lambda x, y: pd.merge(x, y, how='inner', on="Date"), log_return_df_list)
        return merged_df.cov()

    def calculate_beta(self, benchmark_ticker):
        """
        This function calculates the beta of the ETF relative to a benchmark ETF.

        Args:
            benchmark_returns (Series): Daily log returns of the benchmark ETF.
        
        Returns:
            Float: Beta of the ETF relative to the benchmark.
        """
        benchmarkETF = ETF(benchmark_ticker)
        covariance = np.cov(self.log_returns, benchmarkETF.log_returns)[0, 1]
        benchmark_variance = np.var(benchmarkETF.log_returns)
        beta = covariance / benchmark_variance
        return beta

    def summary(self, confidence_level=0.95):
        """
        This function generates a summary of risk metrics.

        Args:
            vconfidence_level (float): The confidence level for the VaR and ES calculations.
    
        Returns:
            DataFrame with risk metrics summary.
        """
        summary_df = pd.DataFrame({
            'Volatility': self.calculate_volatility(),
            'VaR': self.calculate_VaR(confidence_level),
            'ES': self.calculate_ES(confidence_level)
        })
        return summary_df
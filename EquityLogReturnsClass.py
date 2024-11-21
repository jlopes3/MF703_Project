#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from scipy.stats import norm


# In[2]:



class EquityRiskAnalysis_logreturns:
    def __init__(self, log_returns: pd.DataFrame):
        """
        This function initializes the class with a DataFrame of log returns.

        Args:
            log_returns (DataFrame): dataframe with the dates as index and columns as tickers with log returns.
        
        Returns:
            None    
        """
        self.log_returns = log_returns
   
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
        This function calculates the Value at Risk (VaR) at a given confidence level.

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
        This function calculates the Expected Shortfall (ES) at a given confidence level.

        Args:
            confidence_level (float): The confidence level for the ES calculation.
        
        Returns:
            Series with ES of each asset.
        """
        VaR = self.calculate_VaR(confidence_level)
        tail_losses = self.log_returns[self.log_returns < -VaR]
        return -tail_losses.mean()

    def correlation_matrix(self):
        """
        This function calculates the correlation matrix of the returns.

        Returns:
            DataFrame with correlation matrix.
        """
        return self.log_returns.corr()
   
    def covariance_matrix(self):
        """
        This function calculates the covariance matrix of the returns.

        Returns:
            DataFrame with covariance matrix.
        """
        return self.log_returns.cov()

    def summary(self, confidence_level=0.95):
        """
        This function generates a summary of risk metrics.

        Args:
            confidence_level (float): The confidence level for the VaR and ES calculations.
    
        Returns:
            DataFrame with risk metrics summary.
        """
        summary_df = pd.DataFrame({
            'Volatility': self.calculate_volatility(),
            'VaR': self.calculate_VaR(confidence_level),
            'ES': self.calculate_ES(confidence_level)
        })
        return summary_df
    
    def cov_decomposition(self):
        """
        This function generates the eigenvalue decomposition of the covariance matrix.
        
        Returns:
            A tuple of NumPy arrays the first being Eigenvalues, and the second being their corresponding Eigenvectors.
        """
        cov = self.covariance_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvectors = np.transpose(eigenvectors)
        return eigenvalues, eigenvectors
    
    def percent_variation_eigenvalues(self):
        """
        This function generates the weights of each PC from the eigenvalue decomposition.

        Returns:
            A NumPy array of normalized eigenvalues.
        """
        eigenvalues, __ = self.cov_decomposition()
        weighted_eigenvalues = eigenvalues / eigenvalues.sum()
        return weighted_eigenvalues

    def significant_eigenvalues(self, variation = .9):
        """
        This function returns the amount of eigenvalues that are required to make up the percent of variation stated.

        Args:
            variation (float): The percent of variation for the PCA decomposition to explain.

        Return:
            A tuple of the amount of eigenvalues followed by a NumPy array of the eigenvalues themselves.
        
        """
        eig = self.percent_variation_eigenvalues()
        percent_var = 0
        i = 0
        while percent_var < variation:
            percent_var += eig[i]
            i+= 1
        return (i, eig[0:i])

    def significant_eigenvectors(self, variation = .9):
        """
        This function returns the amount of eigenvectors that are required to make up the percent of variation stated.

        Args:
            variation (float): The percent of variation for the PCA decomposition to explain.

        Return:
            A NumPy array of the eigenvectors corresponding to the significant eigenvalues.
        
        """
        num, ___ = self.significant_eigenvalues()
        ___, eigenvectors = self.cov_decomposition()
        return eigenvectors[0:num]


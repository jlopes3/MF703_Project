import numpy as np
import pandas as pd
from scipy.optimize import minimize

class WeightOptimizer:
    def __init__(self, log_returns: pd.DataFrame, risk_free_rate: float = 0.0):
        """
        Initialize the optimizer with asset log returns and risk-free rate.
        """
        self.log_returns = log_returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = log_returns.mean()
        self.cov_matrix = log_returns.cov()
    
    def portfolio_performance(self, weights: np.ndarray):
        """
        Calculate portfolio performance metrics: return, risk, and Sharpe ratio.
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        return portfolio_return, portfolio_risk, sharpe_ratio

    def optimize_weights(self, target_return: float = None):
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
            # Minimize risk for a target return
            result = minimize(self._portfolio_risk, initial_weights, args=args, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        else:
            # Maximize Sharpe ratio
            result = minimize(self._negative_sharpe_ratio, initial_weights, args=args, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise ValueError("Optimization failed:", result.message)
        
        return result.x  # Optimal weights

    def _portfolio_risk(self, weights: np.ndarray, *args):
        """
        Calculate portfolio risk (standard deviation).
        """
        _, cov_matrix, _ = args
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def _negative_sharpe_ratio(self, weights: np.ndarray, *args):
        """
        Calculate negative Sharpe ratio for optimization.
        """
        mean_returns, cov_matrix, risk_free_rate = args
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_risk

    def summary(self, weights: np.ndarray):
        """
        Generate a summary of the portfolio performance.
        """
        portfolio_return, portfolio_risk, sharpe_ratio = self.portfolio_performance(weights)
        return {
            "Weights": weights,
            "Expected Return": portfolio_return,
            "Risk (Std Dev)": portfolio_risk,
            "Sharpe Ratio": sharpe_ratio
        }
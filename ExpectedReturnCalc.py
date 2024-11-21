import numpy as np
import pandas as pd

class ExpectedReturnsCalculator:
    def __init__(self, equities_data, bonds_data, futures_data, market_return=None, risk_free_rate=None):
        """
        Initialize the Expected Returns Calculator.
        """
        self.equities_data = equities_data
        self.bonds_data = bonds_data
        self.futures_data = futures_data
        self.market_return = market_return
        self.risk_free_rate = risk_free_rate

    def calculate_equities_returns(self, method="historical", beta_column=None, dividend_yield_column=None, growth_rate_column=None):
        """
        Calculate expected returns for equities using specified method.
        """
        if method == "historical":
            return self.equities_data.mean(axis=0)  # Average returns
        elif method == "capm":
            if self.market_return is None or self.risk_free_rate is None or beta_column is None:
                raise ValueError("CAPM requires market return, risk-free rate, and beta data.")
            betas = self.equities_data[beta_column]
            return self.risk_free_rate + betas * (self.market_return - self.risk_free_rate)
        elif method == "ddm":
            if dividend_yield_column is None or growth_rate_column is None:
                raise ValueError("DDM requires dividend yield and growth rate data.")
            dividend_yields = self.equities_data[dividend_yield_column]
            growth_rates = self.equities_data[growth_rate_column]
            return dividend_yields + growth_rates
        else:
            raise ValueError(f"Unknown method {method}")

    def calculate_bonds_returns(self, method="ytm", ytm_column=None):
        """
        Calculate expected returns for bonds using specified method.
        """
        if method == "ytm":
            if ytm_column is None:
                raise ValueError("YTM method requires YTM column.")
            return self.bonds_data[ytm_column]
        elif method == "historical":
            return self.bonds_data.mean(axis=0)  # Average returns
        else:
            raise ValueError(f"Unknown method {method}")

    def calculate_futures_returns(self, method="roll_yield", spot_column=None, futures_column=None):
        """
        Calculate expected returns for futures using specified method.
        """
        if method == "roll_yield":
            if spot_column is None or futures_column is None:
                raise ValueError("Roll yield requires spot and futures price data.")
            spot_prices = self.futures_data[spot_column]
            futures_prices = self.futures_data[futures_column]
            return (spot_prices - futures_prices) / futures_prices
        elif method == "historical":
            return self.futures_data.mean(axis=0)  # Average returns
        else:
            raise ValueError(f"Unknown method {method}")

    def calculate_all_returns(self, equities_method="historical", bonds_method="ytm", futures_method="roll_yield", **kwargs):
        """
        Calculate expected returns for all asset classes.
        """
        equities_returns = self.calculate_equities_returns(method=equities_method, **kwargs)
        bonds_returns = self.calculate_bonds_returns(method=bonds_method, **kwargs)
        futures_returns = self.calculate_futures_returns(method=futures_method, **kwargs)

        return pd.DataFrame({
            "Equities": equities_returns,
            "Bonds": bonds_returns,
            "Futures": futures_returns
        })

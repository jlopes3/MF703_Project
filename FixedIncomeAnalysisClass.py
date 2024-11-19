# %%
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
# %%
class Treasuries:
    def __init__(self, ytm, maturity_years, face_value = 100, frequency=2, issue_date=None):
        """
        Initializes a Treasury object.

        Assumes treasuries are issued at par and are compounded semi-annually

        Parameters:
            face_value (float): The bond's face value.
            ytm (float): Current market yield to maturity of the bond.
            maturity_years (float): Years until maturity.
            frequency (int): Coupon payment frequency and compounding frequency per year(default is 2 for semi-annual).
            issue_date (datetime): The bond's issue date.
        """
        self.face_value = face_value
        self.ytm = ytm/100 #yields are quoted in percentages on Bloomberg
        self.maturity_years = maturity_years
        self.frequency = frequency
        self.issue_date = issue_date if issue_date else datetime.today()
        self.maturity_date = self.issue_date + timedelta(days=maturity_years * 365)
        self.periods = int(self.maturity_years * self.frequency)
    
    def cash_flows(self):
        """
        Calculates the cash flows of the bond.

        Returns:
            np.array: Cash flows array for each period until maturity.
        """
        periods = self.periods
        if periods == 1:
            return np.array([self.face_value])

        coupon_payment = self.face_value * self.ytm / self.frequency #since we assume treasuries are issued at par, the coupon payment = the yield to maturity
        cash_flows = np.full(periods, coupon_payment)
        cash_flows[-1] += self.face_value  # Adding face value at maturity
        return cash_flows
    
    def discount_factors(self):
        """ returns the discount factors as a numpy array"""
        return np.array([(1+self.ytm/self.frequency)**(-i) for i in range(1,self.periods + 1)])

    def price(self):
        """
        Calculates the bond's price. Since we assume Treasuries are issued at par, the price should be 100

        Returns:
            float: Price of a bond rounded to 2 decimal places.
            example: 98.25 -> $98.25 / 100 face
        """
        cash_flows = self.cash_flows()

        discounts = self.discount_factors()
        price = round(discounts@cash_flows, 2)

        return price

    def modified_duration(self):
        """
        Calculates the bond's modified duration.

        Returns:
            float: Modified duration.
        """

        der_discount = np.array([-i/self.frequency *(self.ytm/self.frequency)**(-i-1) for i in range(1,self.periods + 1)])
        fprime = self.cash_flows() @ der_discount

        return - fprime / self.price()
    
    def macaulay_duration(self):
        """
        Calculates the Macaulay duration of the bond.

        Returns:
            float: Macaulay duration.
        """

        return  (1+self.ytm/self.frequency) * self.modified_duration()

    def convexity(self):
        """
        Calculates the convexity of the bond.

        Returns:
            float: Convexity.
        """
        
        der_2_discount = np.array([ i*(i+1)/(self.frequency)**2 *(1+self.ytm/self.frequency)**(-i-2) for i in range(1,self.periods + 1)])
        f_2_prime = der_2_discount @ self.cash_flows()
        
        return f_2_prime/self.price()

    def value_at_risk(self, confidence_level=0.95):
        """
        Estimates the value at risk (VaR) of the bond based on duration and convexity.

        Parameters:
            confidence_level (float): The confidence level for VaR calculation.

        Returns:
            float: Estimated VaR for the bond.
        """
        modified_duration = self.modified_duration()
        convexity = self.convexity()
        yield_shock = np.percentile(np.random.normal(0, 0.01, 10000), 1 - confidence_level)
        price_change = -self.price() * (modified_duration * yield_shock + 0.5 * convexity * yield_shock**2)
        return price_change

    def election_cycle_risk_analysis(self, election_date):
        """
        Analyzes the bond's performance over an election cycle, six months before and 
        two weeks post-election.

        Parameters:
            election_date (datetime): The date of the election.

        Returns:
            dict: A dictionary with metrics calculated for the election cycle.
        """
        start_period = election_date - timedelta(days=6*30)
        end_period = election_date + timedelta(days=14)
        
        if self.issue_date > end_period or self.maturity_date < start_period:
            return {"error": "Bond is not active during the specified election cycle"}

        ytm = self.ytm
        mod_duration = self.modified_duration()
        convexity = self.convexity()
        election_vasr = self.value_at_risk()
        
        analysis_results = {
            "yield_to_maturity": ytm,
            "modified_duration": mod_duration,
            "convexity": convexity,
            "value_at_risk": election_vasr,
            "start_period": start_period,
            "end_period": end_period
        }

        return analysis_results
    
    def calculate_beta(asset_returns, market_returns):
         """
        Calculate the beta of an asset relative to the market.

        Parameters:
            asset_returns (np.array): Array of historical returns of the asset.
            market_returns (np.array): Array of historical returns of the market.

        Returns:
            float: The beta of the asset.
        """
        covariance_matrix = np.cov(asset_returns, market_returns)
        covariance = covariance_matrix[0, 1]
        market_variance = np.var(market_returns, ddof=1)
        beta = covariance / market_variance
        return beta


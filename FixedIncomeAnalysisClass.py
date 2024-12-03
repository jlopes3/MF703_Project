# %%
import numpy as np
import pandas as pd
import scipy as sci
from datetime import datetime, timedelta
# %%
def election_day(year):
    date = pd.Timestamp(year=year, month=11, day=1)
    return date + pd.offsets.Week(weekday=1)

class Treasuries:
    def __init__(self, ytm, maturity_years, face_value = 100, frequency=2, issue_date=datetime.today()):
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
        self.issue_date = issue_date
        self.maturity_date = self.issue_date + timedelta(days=maturity_years * 365)
        self.periods = int(self.maturity_years * self.frequency)
        self.cash_flow_dates = pd.Series([issue_date + timedelta(weeks=26*i) for i in range(1, self.periods + 1)])
    
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

    def election_cycle_risk_analysis(self, election_date: int):
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
    
    def times_till_coupon(self, current_date):
        """Creating a numpy array to get the values of time till coupon payments given a new time
            Returns time till maturity in number of days till coupon/360
        """
        time_till_next_coupon = self.cash_flow_dates.apply( lambda x : (x - current_date))
        time_till_next_coupon = time_till_next_coupon.apply(lambda x: x.days)
        time_till_next_coupon = time_till_next_coupon.apply(lambda x: max(x, 0))
        return np.array(time_till_next_coupon/360)
    
    def interpolated_yield_curve(self, S: pd.Series, T):
        """Returning of yield curve interpolation method given a yield curve (pandas series) and time of maturity T."""
        icurve = sci.interpolate.CubicSpline(x= S.index, y = S)

        return icurve(T)
    
    def spot_rates(self, S: pd.Series):
        """Bootstrapping spot rates from a yield curve interpolation"""

        # Computing yields from the interpolated yield curve
        tenors = np.arange(.5, 30 + .5, .5)

        yields = self.interpolated_yield_curve(S, tenors)/100

        # Bootstrapping spot rates
        s_rates = np.array([yields[0]])
        discount_factors = (1+s_rates/2)**(-1)

        for i in range(1, len(tenors)):
            y = yields[i]

            first_part = 1 - y/2*discount_factors.sum()
            second_part = (1+y/2)**(-1)

            discount_factors = np.append(discount_factors, first_part * second_part)

            if first_part <= 0:
                raise ValueError(f"Negative or zero discount factor at tenor {tenors[i]}. Check yield inputs.")
            
            s_rates = np.append(s_rates, 2*discount_factors[i]**(-1/(2*tenors[i])) - 2)
        
        return s_rates
    
    def interpolated_spot_curve(self, S:pd.Series, T):
        """Interpolating the Spot Curve given a pandas series containing yields, and a new time T"""
        s_rates = self.spot_rates(S= S)
        tenors = np.arange(.5 , 30+.5, .5)
        icurve = sci.interpolate.CubicSpline(x= tenors, y = s_rates)

        return icurve(T)

    
    def new_price(self, S:pd.DataFrame):
        """Calculating the daily price of a bond given a new yield curve.
        S - S.index : datetime objects or pandas timeseries/timestamps
        Returns a pandas series with price data and indices of the dates
        """

        #initializing an empty series for the for loop
        price_array = np.array([])



        #calculating the price per row
        for i in range(len(S.index)):
            #getting the row for the dataframe
            row = S.iloc[i,:]

            # Calculating Tenors for coupon payments
            ## Remember tenors are (number of days till next coupon/360)
            tenors = self.times_till_coupon(current_date=row.name)
            tenors = tenors[tenors>0]


            ## checking if tenors is empty. if it is, the bond has matured, and is equal to the face value
            if len(tenors) == 0:
                price_array = np.append(price_array, self.face_value)
                continue

            #getting remaining coupons
            coupons = self.cash_flows()[-len(tenors):]

            # getting spot rates
            ## If there is an error in finding the discount rates, we use the spot curve of the previous day.
            j = i
            s_rates = 0
            while True:
                try:
                    s_rates = self.interpolated_spot_curve(S= S.iloc[j,:], T= tenors)
                except ValueError:
                    j-= 1
                else:
                    break
                if abs(j - i) > 30:
                    j= i
                    break
            
            ## If no spot rate can be found, we go in the reverse direction
            while True:
                try:
                    s_rates = self.interpolated_spot_curve(S= S.iloc[j,:], T= tenors)
                except ValueError:
                    j+=1
                except IndexError:
                    print("Interpolated curves cannot support discount rates")
                    break
                else:
                    break
            
            #setting empty discount factors array
            discount_factors = np.array([])

            if len(s_rates)!= len(coupons):
                print(s_rates)

            #calculating the discount factors
            for k in range(len(tenors)):
                disc_factor = (1+s_rates[k]/2)**(-2*tenors[k])
                discount_factors = np.append(discount_factors, disc_factor)

            #dotting the coupon vector with the discount factors vector to get the new price
            price_array = np.append(price_array, coupons @ discount_factors)

        price_series = pd.Series(price_array, index = S.index)
        return price_series


class Treasury(Treasuries):
    """Creating a class that takes in a Dataframe, and choice of treasury that inherits from the Treasuries class
    
    Paramters:
        maturity_years : float or int
        df: pandas dataframe containing par yield curves
        face_value: int. assumes 100 face
        frequency: float. compounding frequency. assume semi-annual compounding
    
    """
    def __init__(self, maturity_years, df: pd.DataFrame, face_value_ = 100, 
                 frequency_=2):
        
        # collecting data from the dataframe
        ytm_ = df.iloc[0,:][maturity_years]
        i_date = df.index[0]
        self.par_curve = df
        super().__init__(ytm_, maturity_years, face_value_, frequency_, i_date)
    
    def new_price(self):
        """
        Calculate the daily price of the bond using the DataFrame passed during initialization.
        
        Uses the `new_price` method from the parent class but automatically
        uses `self.df` as the DataFrame input.
        
        Returns:
            pd.Series: A Series of bond prices indexed by dates from the DataFrame.
        """
        return super().new_price(self.par_curve)
    
    def open_position(self):
        return self.price()

    def close_position(self):
        """A method to get the full closing position of the bond.
            Accounts for the coupon payment during the period.
            TAKES ABOUT A MINUTE TO RUN since it calls Treasury.new_price()
        """
        position = self.new_price().iloc[-1,:][self.maturity_years]
        if self.maturity_years != .5:
            position += self.cash_flows()[0]
        
        return position 



# %%

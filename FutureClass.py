# 
# Title: Future Class
# Name: Aiden Perkins
# Email address: ajp15@bu.edu
# Description: Class for generic futures price analysis
# 
#
import pandas as pd
from scipy.stats import f

class future:
    
    def __init__(self,filename):
        """ Initializes a Future class given a csv filename
        """
        self.symbol = filename[:-5]
        
        self.df = pd.read_excel(filename,parse_dates=True,date_format="%m/%d%/%Y",index_col="Date")
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df[self.df['PX_LAST'].notna()]
        
        self.df["daily_returns"] = self.df["PX_LAST"].pct_change(1,fill_method=None)
        self.stdev = self.df["daily_returns"].std()
        
        self.all_returns = self.df["daily_returns"]
        self.election_returns = self.df.loc[
            (self.df.index.year % 4 == 0) & 
            (((self.df.index.month >= 5) & (self.df.index.month < 11)) | 
             ((self.df.index.month == 11) & (self.df.index.day <= 21))),
            'daily_returns' ]
        self.non_election_returns = self.df.loc[
            ~((self.df.index.year % 4 == 0) & 
              (((self.df.index.month >= 5) & (self.df.index.month < 11)) | 
               ((self.df.index.month == 11) & (self.df.index.day <= 21)))),'daily_returns']
    
    def correlation(self, other, period = "all"):
        """ Returns the correlation of one Future returns and 'other'
            input other: Future class or Pandas series of returns for a different asset
            input period: timeframe to compare (all, election, non_election)
        """
        if type(other) == future:
            if(period == "all"):
                ret = self.df['daily_returns'].corr(other.df['daily_returns'])
                return ret
            if (period == 'election'):
                ret = self.election_returns.corr(other.election_returns)
                return ret
            else:
                ret = self.non_election_returns.corr(other.non_election_returns)
                return ret
        else:
            ret = self.df['daily_returns'].corr(other)
            return ret
        
    def election_var_F_test(self):
        var_election = self.election_returns.var(ddof=1)
        var_non_election = self.non_election_returns.var(ddof=1)
        
        f_stat = max(var_election, var_non_election) / min(var_election, var_non_election)
        
        df1 = len(self.election_returns)
        df2 = len(self.non_election_returns)
        
        p_value = 1 - f.cdf(f_stat,df1,df2)
        
        return (f_stat, p_value)
        
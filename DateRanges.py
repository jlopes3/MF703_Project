import datetime as dt
import pandas as pd

def election_day(year):
    date = pd.Timestamp(year=year, month=11, day=1)
    return date + pd.offsets.Week(weekday=1)

class ElectionYears:
    """Getting information for date ranges during election years"""
    def __init__(self):
        self.start_date = dt.datetime(year = 1992, month = 1, day = 1)
        self.o_range = 2024 - self.start_date.year #overall range
        self.num_ranges = self.o_range // 4

    def ranges(self):
        """Getting ranges for each year that we are analyzing.
        returns dictionary with keys = year and values that are lists.
            list[0] = period_start_date
            list[1] = period_end_date
        """

        r_ranges = {} #returning ranges for years. 
        # values of r_ranges will be lists. index 1 returns the start date. Index 

        for i in range(self.num_ranges):
            year = self.start_date.year + 4*i
            e_day = election_day(year).to_pydatetime()
            start = dt.datetime(year = year, month = e_day.month - 6, day = e_day.day)
            end = e_day + dt.timedelta(days = 14)

            r_ranges[str(year)] = [start, end]

        return r_ranges

class NonElectionYears(ElectionYears):
    """Getting information for date ranges during non-election years"""
    def __init__(self, years_after_election =1):
        if years_after_election not in [1,3]:
            raise ValueError("Input must be either 1 or 3")
            print("You entered" + str(years_after_election))
        self.start_date = dt.datetime(year = 1992 + years_after_election, month = 1, day = 1)
        self.o_range = 2024 - self.start_date.year #overall range
        self.num_ranges = self.o_range // 4

e_year_ranges = ElectionYears().ranges()
ne_year_ranges = NonElectionYears().ranges()
import pandas as pd
import datetime
import os
from os.path import isfile, join


class Market:

    def __init__(self):
        self.market = dict()
        # Load stock information from file
        path = "stock_info"
        files = [(path, f) for f in os.listdir(path) if isfile(join(path, f))]
        # Save price of each stock
        for path, file_name in files:
            stock_name = os.path.splitext(file_name)[0]
            self.market[stock_name] = pd.read_csv(join(path, file_name))

        # Current date of the environment
        # User only have access to the price up to this value
        self.date = datetime.datetime(2020, 5, 11)

    def next_day(self):
        self.date += datetime.timedelta(days=1)

    def get_date(self):
        return self.date

    def look_up_price(self, stock_name, date):
        """
        Return a dataframe containing prices of given stock name and date
        :param stock_name:
        :param date:
        :return:
        """
        df = self.market[stock_name]
        df = df[pd.to_datetime(df['Date']) <= self.date]
        if isinstance(date, list):
            return df[pd.to_datetime(df['Date']).isin(date)]
        return df[pd.to_datetime(df['Date']) == date]

    def get_history(self, stock_name):
        df = self.market[stock_name]
        return df[pd.to_datetime(df['Date']) <= self.date]

    def get_cur_price(self):
        for stock_name in self.market.keys():
            print(stock_name)
            stock_info = self.look_up_price(stock_name, self.date)
            if not stock_info.empty:
                print(stock_info.to_string(index=False))


m = Market()
m.get_cur_price()


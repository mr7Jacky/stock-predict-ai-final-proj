from market import Market
import numpy as np


class Account:

    def __init__(self):
        self.purchasing_pwr = 0
        self.property = self.purchasing_pwr
        self.stock = {}
        self.market = Market()

    def add_fund(self, amount):
        self.purchasing_pwr += amount

    def buy(self, stock_name, volume):
        cur_date = self.market.get_date()
        price = self.market.look_up_price(stock_name, cur_date)['Open'][0] # TODO-what price to get
        if price * volume > self.purchasing_pwr:
            print("[ERROR] Not enough purchasing power")
        else:
            self.purchasing_pwr -= price * volume
            self.stock[stock_name] = (price, volume)

    def sell(self, stock_name, volume):
        try:
            cur_date = self.market.get_date()
            price = self.market.look_up_price(stock_name, cur_date)['Close'][0]  # TODO-what price to get
            buying_rate, holding_volume = self.stock[stock_name]
            if holding_volume < volume:
                print("[ERROR] Not enough remaining volume")
            else:
                holding_volume -= volume
                self.purchasing_pwr += (price - buying_rate) * volume
        except KeyError:
            print("[ERROR] Haven't bought stock")

    def update_property(self):
        self.property = self.purchasing_pwr
        self.property += np.sum([price * volume for price, volume in self.stock])

from market import Market
import numpy as np


class Account:
    """
    Tis class is a virtual Account used for selling and buying stock from market.
    It also provide calculation and statics that simulate the real stock selling and buying environment
    """
    def __init__(self):
        """
        Initialization
        """
        # The money that could be spent
        self.purchasing_pwr = 0
        # The total money holds
        self.property = self.purchasing_pwr
        # All the stocks holds
        self.stock = {}
        # The virtual stock market
        self.market = Market()

    def add_fund(self, amount):
        """
        Add money that could be spent
        :param amount: the amount to add
        """
        self.purchasing_pwr += amount

    def buy(self, stock_name, volume):
        """
        Buy action
        Select a stock and buy certain volume
        The money spend would be subtract from purchasing power
        :param stock_name: the stock to buy
        :param volume: the amount to buy
        """
        cur_date = self.market.get_date()
        price = self.market.look_up_price(stock_name, cur_date)['Open'][0] # TODO-what price to get
        if price * volume > self.purchasing_pwr:
            print("[ERROR] Not enough purchasing power")
        else:
            self.purchasing_pwr -= price * volume
            self.stock[stock_name] = (price, volume)

    def sell(self, stock_name, volume):
        """
        Sell action
        Sell a given stock from holding stocks,
        The prices different times the volume would be add back to purchasing power
        :param stock_name: the stock to sell
        :param volume: the amount to sell
        """
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
        """
        Update the total money holds
        """
        self.property = self.purchasing_pwr
        self.property += np.sum([price * volume for price, volume in self.stock])

# https://github.com/gustavovargas/stocktradingrl
import gym
from gym import spaces
import numpy as np
import math


class CustomEnv(gym.Env):
    """
    A stock trading environment based on the one made up by Adam King
    (github.com/notadamking/Stock-Trading-Environment) for OpenAI gym
    """

    def __init__(self, train_data, eval_data, len_obs=20, len_window=100, init_balance=1000,
                 action_list=(-0.30, -0.20, 0, 0.20, 0.30)):
        """
        initialize the environment:
            specify details of the environment
        :param train_data: the dada used to train the agent
        :param eval_data: the data used for evaluation
        :param len_obs: length of observation of the agent
        :param len_window: length of total space that can be operated by agent
        :param init_balance: initial balance of agent
        :param action_list: list of potential actions
        """
        super(CustomEnv, self).__init__()
        # changing the train_data to prediction make by nn
        self.train_data = train_data
        self.eval_data = eval_data
        self.init_balance = init_balance

        self.len_obs = len_obs
        self.len_window = len_window
        self.action_list = action_list
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box(-5000, 5000, shape=(self.len_obs, 1), dtype=np.float32)
        self.profits_rec = []
        self.list_nw = []

    def reset(self, train_data=True, batch_size=1, overlap=20):
        """
        reset the enviroment to be ready for training
        :param train_data: if reset the train data
        :param batch_size: number of agents to be trained synchronously
        :param overlap: overlap of train data
        """
        # Reset the state of the environment to an initial state
        self.batch_size = batch_size
        self.balance = [self.init_balance for i in range(batch_size)]
        self.net_worth = [self.init_balance for i in range(batch_size)]
        self.shares_held = [0 for i in range(batch_size)]
        self.cost_basis = [0 for i in range(batch_size)]
        self.total_shares_sold = [0 for i in range(batch_size)]
        self.total_sales_value = [0 for i in range(batch_size)]
        self.list_nw = np.array([])

        # Set the current step to a random point within the dataframe
        if train_data:
            idx = np.random.randint(self.len_obs, len(self.train_data) - self.len_window, (batch_size,))
            self.prices = np.array([self.train_data[i - self.len_obs:i + self.len_window, 0] for i in idx])
            self.returns = np.array([self.train_data[i - self.len_obs:i + self.len_window, 1] for i in idx])
        else:
            idx = np.arange(self.len_obs, len(self.eval_data) - self.len_window, overlap)
            self.prices = np.array([self.eval_data[i - self.len_obs:i + self.len_window, 0] for i in idx])
            self.returns = np.array([self.eval_data[i - self.len_obs:i + self.len_window, 1] for i in idx])
        self.posit_window = 0
        return self._next_observation()

    def _next_observation(self):
        """
        Get data points for the last 20 days
        """
        frame = self.returns[:, self.posit_window:self.posit_window + self.len_obs]
        return frame

    def step(self, action):
        """
        Execute one time step within the environment:
        :param action: the action to be taken
        """
        self._take_action(action)
        self.posit_window += 1

        if self.posit_window == 1:
            reward = [0 for i in range(self.batch_size)]
        else:
            reward = (self.list_nw[:, -1] / self.list_nw[:, -2]) - 1
        done = (self.posit_window == self.len_window)
        obs = self._next_observation()

        return obs, reward, done, {}

    def buy(self, current_price, idx, percentage):
        """
        Buy a certain amount of stock holdings
        :param current_price: current price of stock
        :param idx: index of agent
        :param pertencage: percentage of stock to buy
        """
        total_possible = self.balance[idx] / current_price[idx]
        shares_bought = math.floor(total_possible * percentage)
        prev_cost = self.cost_basis[idx] * self.shares_held[idx]
        additional_cost = shares_bought * current_price[idx]

        self.balance[idx] -= additional_cost
        self.cost_basis[idx] = (prev_cost + additional_cost) / (self.shares_held[idx] + shares_bought + 1)
        self.shares_held[idx] += shares_bought

    def sell(self, current_price, idx, percentage):
        """
        Sell a certain amount of stock holdings
        :param current_price: current price of stock
        :param idx: index of agent
        :param pertencage: percentage of stock to sell
        """
        try:
            shares_sold = math.floor(self.shares_held[idx] * percentage)
        except:
            print(self.shares_held)
            print(idx)
        self.balance[idx] += shares_sold * current_price[idx]
        self.shares_held[idx] -= shares_sold
        self.total_shares_sold[idx] += shares_sold
        self.total_sales_value[idx] += shares_sold * current_price[idx]

    def _take_action(self, actions):
        """
        take the next action:
        :param actions: potential actions
        """
        current_price = self.prices[:, self.posit_window + self.len_obs - 1]

        mid = len(self.action_list) // 2
        for idx, act in enumerate(actions):
            if act > mid:
                # Buy balance in shares
                self.buy(current_price, idx, self.action_list[act])
            elif act == mid:
                pass
            elif act < mid:
                # Sell shared held
                self.sell(current_price, idx, np.abs(self.action_list[act]))
        self.net_worth = self.balance + (self.shares_held * current_price)
        if self.posit_window == 0:
            self.list_nw = np.array(self.net_worth).reshape(-1, 1)
        else:
            self.list_nw = np.concatenate([self.list_nw, np.array(self.net_worth).reshape(-1, 1)], axis=1)

        if self.shares_held == [0 for i in range(len(self.shares_held))]:
            self.cost_basis = [0 for i in range(self.batch_size)]

    def result(self, days_per_year=252):
        """
        returns the (annual) result of training:
        :param days_per_year: days of operation in a year(cycle)
        :return:
        cagr: compound annual growth rate this time
        ann_vol: annual volume
        """
        div = self.list_nw[:, -1] / self.list_nw[:, 0]
        exp = (days_per_year / self.len_window)
        cagr = (div ** exp) - 1
        ann_vol = ((self.list_nw[:, 1:] / self.list_nw[:, :-1]) - 1).std(axis=1) * (days_per_year ** 0.5)  # !!
        return cagr, ann_vol

    def render(self, ep, close=False):
        """
        render (print) the results:
        :param ep: the number of epoch
        :param close: if close the simulation
        """
        profit = (self.net_worth - self.init_balance) / self.init_balance
        mean_profit = np.mean(profit)
        vol_profit = np.std(profit)
        # Render the environment to the screen
        print(f'\nep {ep} ')
        print(f'Profit: {mean_profit * 100}%')
        print(f'Std: {vol_profit}')

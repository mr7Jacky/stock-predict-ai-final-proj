from gym.envs.registration import register
from stock_env.custom_env import *
register(
    id='StockEnv-v0',
    entry_point='stock_env:CustomEnv',
    max_episode_steps=2000,
)
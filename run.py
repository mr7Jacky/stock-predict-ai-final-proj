import gym
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


import sys
sys.path.insert(1, './agent')
sys.path.insert(1, './price_pred')

import stock_env
from QLAgent import QNAgent
from helper import *

def run(seed, variable):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
    name_file_data = 'AAPL.csv'
    name_project = 'AAPL_batch'
    env_name = 'StockEnv-v0'
    name_model_weights = 'trading_weights'
    load_model = False #if we load model
    epoch_to_load = 10000 # num of epoch to load when loading previous data

    num_episodes = 50 # training epoches
    render_p = 10 # print frequency
    init_balance = 1000 # initial fund agent have

    len_obs = 60 # observation length, number of days the agent look back
    len_window = 100 # num of times trained each time
    interval = 1 # interval of validation
    overlap = 20 # overlapping between two consecutive training data 
    batch_size = 1000 
    a = [i/10 for i in range(-8,9,1)]
#     print(tuple(a))
    action_list=tuple(a)
    path_models, path_stats = check_directories(name_project, len_obs, len_window)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Read data you want to use for trading
    train_data, test_data = get_data(f'./data/{name_file_data}')
    # Create an instant
    env = gym.make(env_name, train_data=train_data, eval_data=test_data, 
                   len_obs=len_obs, len_window=len_window, init_balance=init_balance,
                   action_list=action_list)
    env.seed(seed)
#     print(f'Observation space: {env.observation_space}')
#     print(f'Action space: {env.action_space}')
    # Create an agent
    agent = QNAgent(env, discount_rate=0.5, learning_rate=variable, epsilon=0.01)
    # Load model
    if load_model:
        agent.load(path_models + name_model_weights + f'_{epoch_to_load}')
        train_statistics = pd.read_csv(path_stats + 'train.csv')
        test_statistics = pd.read_csv(path_stats + 'test.csv')
        init_ep = epoch_to_load
    else:
        init_ep = 0
        train_statistics = pd.DataFrame()
        test_statistics = pd.DataFrame()
    # Decay of epsilon
    initial_epsilon = 0.1
    delta_epsilon = 0.01
    worth = []
    losses = []
    for ep in range(init_ep, num_episodes):
        _, _, loss = get_performance(env, agent, train_data=True, training=True, batch_size=batch_size)
#         if (ep % render_p) == 0:
#             env.render(ep)
    #     if (ep % interval_epsilon) == 0:
    #         agent.epsilon -= delta_epsilon
        worth.append(env.net_worth)
        losses = np.concatenate((losses,loss))
        if (ep % interval == 0) and not((load_model==True) and (ep == epoch_to_load)):
            agent.model.save_weights(path_models)

            overlap = overlap
            results_train = np.empty(shape=(0, 3))
            results_test = np.empty(shape=(0, 3))

            size_test = ((len(env.eval_data)-env.len_obs-env.len_window) // overlap)+1
            cagr_train, vol_train, _ = get_performance(env, agent, train_data=True, training=False, batch_size=size_test)
            results_train = np.array([np.tile(ep, size_test), cagr_train, vol_train]).transpose()

            cagr_test, vol_test, _ = get_performance(env, agent, train_data=False, training=False, overlap=overlap, batch_size=size_test)
            results_test = np.array([np.tile(ep, size_test), cagr_test, vol_test]).transpose()
    return losses, worth
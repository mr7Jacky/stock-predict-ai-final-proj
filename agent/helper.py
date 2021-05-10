import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import os
import argparse

import sys
sys.path.insert(1, '../')
import stock_env

from QLAgent import QNAgent                                                                

sns.set()  # just for better images


def get_data(path, index_col=0, train_pct=0.7):
    """
    :param train_pct:
    :param index_col:
    :param path:
    :return:
    """
    data = pd.read_csv(path, index_col=index_col, parse_dates=True, header=0)
    data = data[['Close']]
    data = pd.concat([data, data.pct_change()], axis=1).iloc[1:]
    data.columns = ['prices', 'returns']
    
    sep = np.floor(len(data) * train_pct).astype('int')
    train_data = data.iloc[:sep, :]
    test_data = data.iloc[sep:, :]
    return train_data.values, test_data.values


def get_performance(env, agent, train_data=True, training=False, batch_size=1, overlap=20):
    state = env.reset(train_data=train_data, batch_size=batch_size, overlap=overlap)
    done = False
    while not done:
        action = agent.get_action(state, use_random=True)
        next_state, reward, done, info = env.step(action)
        if training:
            agent.train((state, action, next_state, reward, done))
        state = next_state
    cagr, vol = env.result()
    return cagr, vol


def check_directories(name_project: str, len_obs:str, len_window:str):
    name_project = f'{name_project}_obs{len_obs}_window{len_window}'
    directories = [f'projects/{name_project}' for dir in ['saved_models', 'statistics']]
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return directories

def plot_stocks_trading_performance(data, color='royalblue', alpha=0.5, s=12, acc_title=''):
    plt.scatter(data[:, 2] * 100, data[:, 1] * 100, alpha=alpha, s=s, color=color)
    plt.xlabel('Volatility')
    plt.ylabel('CAGR')
    plt.title(f'RL Trading - Episode {ep} {acc_title}')
    plt.xlim(0, 50)
    plt.ylim(-50, 100)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.close()
    plt.cla()
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Trading with Q Learning')
    parser.add_argument('-name_file_data', default='AAPL.csv', type=str)
    parser.add_argument('-name_project', default='AXP_batch', type=str)
    parser.add_argument('-env_name', type=str, default='StockEnv-v0')
    parser.add_argument('-num_episodes', type=int, default=10)
    parser.add_argument('-len_obs', type=int, default=50)
    parser.add_argument('-len_window', type=int, default=100)
    parser.add_argument('-interval', type=int, default=1000)
    parser.add_argument('-load_model', type=bool, default=False)
    parser.add_argument('-epoch_to_load', type=int, default=10000)
    parser.add_argument('-name_model_weights', type=str, default='trading_weights')
    parser.add_argument('-overlap', type=int, default=20)

    args = parser.parse_args()
    path_models, path_imgs, path_stats = check_directories(args.name_project, args.len_obs, args.len_window)
    
    np.random.seed(42)
    # Read data you want to use for trading
    train_data, test_data = get_data(f'../data/{args.name_file_data}')

    # Create an instant
    env = gym.make(args.env_name, train_data=train_data, eval_data=test_data, len_obs=args.len_obs, len_window=args.len_window, init_balance=1000)
    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')

    # Create an agent
    agent = QNAgent(env)

    # Load model
    if args.load_model:
        agent.load(path_models + args.name_model_weights + f'_{args.epoch_to_load}')
        train_statistics = pd.read_csv(path_stats + 'train.csv')
        test_statistics = pd.read_csv(path_stats + 'test.csv')
        init_ep = args.epoch_to_load
    else:
        init_ep = 0
        train_statistics = pd.DataFrame()
        test_statistics = pd.DataFrame()

    for ep in range(init_ep, args.num_episodes):
        get_performance(env, agent, train_data=True, training=True, batch_size=1)
        env.render(ep)

        if (ep % args.interval == 0) and not((args.load_model==True) and (ep == args.epoch_to_load)):
            agent.model.save_weights(path_models + args.name_model_weights + f'_{ep}')

            overlap = args.overlap
            results_train = np.empty(shape=(0, 3))
            results_test = np.empty(shape=(0, 3))

            size_test = ((len(env.eval_data)-env.len_obs-env.len_window) // overlap)+1
            cagr_train, vol_train = get_performance(env, agent, train_data=True, training=False, batch_size=size_test)
            results_train = np.array([np.tile(ep, size_test), cagr_train, vol_train]).transpose()

            cagr_test, vol_test = get_performance(env, agent, train_data=False, training=False, overlap=overlap, batch_size=size_test)
            results_test = np.array([np.tile(ep, size_test), cagr_test, vol_test]).transpose()

            train_statistics = pd.concat([train_statistics, pd.DataFrame(results_train, columns=['epoch', 'cagr','volatility'])])
            train_statistics.to_csv(path_stats+'train.csv', index=False)
            test_statistics = pd.concat([test_statistics, pd.DataFrame(results_test, columns=['epoch', 'cagr','volatility'])])
            test_statistics.to_csv(path_stats+'test.csv', index=False)

            plot_stocks_trading_performance(results_train, path_imgs + f'train_cagr_vol_ep_{ep}',
                                            color='royalblue', acc_title='Train')
            plot_stocks_trading_performance(results_test, path_imgs + f'test_cagr_vol_ep_{ep}',
                                            color='firebrick', acc_title='Test')

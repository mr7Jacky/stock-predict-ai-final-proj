# Third-party library
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

sns.set()  # for better images
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# Build-in library
import os
import argparse
# Custom library
import sys

sys.path.insert(1, '../')
import stock_env
from QLAgent import QNAgent


def get_data(path, index_col=0, train_pct=0.7):
    """
    This function read data set from given path
    :param train_pct: percentage of training data in terms of total number of data
    :param index_col: index of the column
    :param path: path to a file
    :return: training data and testing data
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
    """
    Perform the training and get feedback
    :param env: environment for agent to perform
    :param agent: stock trading agent
    :param train_data: whether using the training data
    :param training: whether to train the model
    :param batch_size: size of batch
    :param overlap: overlapping of data
    :return: training data and testing data
    """
    state = env.reset(train_data=train_data, batch_size=batch_size, overlap=overlap)
    done = False
    loss = []
    while not done:
        action = agent.get_action(state, use_random=True)
        next_state, reward, done, info = env.step(action)
        if training:
            loss.append(agent.train((state, action, next_state, reward, done)))
        state = next_state
    cagr, vol = env.result()
    return cagr, vol, loss


def check_directories(name_project: str, len_obs: str, len_window: str):
    """
    Make the directory for saved models and performance
    :param name_project: name of the project
    :param len_obs: key feature in training for naming
    :param len_window: key feature in training for naming
    :return: a list of directory names.
    """
    name_project = f'{name_project}_obs{len_obs}_window{len_window}'
    directories = [f'projects/{name_project}' for dir in ['saved_models', 'statistics']]
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return directories


def plot_stocks_trading_performance(data, color='royalblue', alpha=0.5, s=12, acc_title=''):
    """
    Function to plot Volatility vs CAGR
    :param data: data to plot
    :param color: plot color
    :param alpha: transparency of plotting
    :param s: the marker size in points
    :param acc_title: title for plot
    """
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

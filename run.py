import gym
import keras 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
import ta

import pickle
import argparse
from datetime import timedelta
import os
import sys
sys.path.insert(1, './price_pred')
sys.path.insert(1, './agent')

from nn import NN
import stock_env
from QLAgent import QNAgent
from helper import *

def load_pretrained(model):
    # Load param
    param = pickle.load(open('pretrained_models/pred/param', 'rb'))
    # Load model from pretrain model
    if model == 'lstm':
        model = keras.models.load_model('pretrained_models/pred/pred_lstm')
    elif model == 'drnn':
        model = keras.models.load_model('pretrained_models/pred/pred_drnn')
    else:
        raise ValueError('Model cannot found.')
    return param, model

def preprocess(data):
    # Preprocess
    close_scaler = RobustScaler()
    scaler = RobustScaler()
    # Datetime conversion
    data['Date'] = pd.to_datetime(data['Date'])
    # Setting the index
    data.set_index('Date', inplace=True)
    # Dropping any NaNs
    data.dropna(inplace=True)
    # Technical Indicators
    # Adding all the indicators
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low",
                                  close="Close", volume="Volume", fillna=True)
    # Dropping everything else besides 'Close' and the Indicators
    data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    # Only using the last 1000 days of data to get a more accurate representation of the current market climate
    data = data.tail(1000)
    close_scaler.fit(data[['Close']])
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return data, close_scaler, scaler

def predict(df, n_per_in, n_per_out, n_features, model, close_scaler, oname):
    # Predicting off of the most recent days from the original DF
    y_hat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features))
    # Transforming the predicted values back to their original format
    y_hat = close_scaler.inverse_transform(y_hat)[0]
    # Creating a DF of the predicted prices
    preds = pd.DataFrame(y_hat, index=pd.date_range(start=df.index[-1] + timedelta(days=1),
                                                    periods=len(y_hat), freq="B"),
                         columns=[df.columns[0]])
    # Number of periods back to plot the actual values
    pers = n_per_in
    # Transforming the actual values to their original price
    actual = pd.DataFrame(close_scaler.inverse_transform(df[["Close"]].tail(pers)),
                          index=df.Close.tail(pers).index,
                          columns=[df.columns[0]]).append(preds.head(1))
    # Printing the predicted prices
    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(actual, label="Actual Prices")
    plt.plot(preds, label="Predicted Prices")
    plt.ylabel("Price")
    plt.xlabel("Dates")
    plt.title(f"Forecasting the next {len(y_hat)} days")
    plt.legend()
    plt.savefig(oname + '_pred.png', dpi=300)
    preds.to_csv(oname + '_pred.csv')

def visualize_training_results(results, oname):
    """
    Plots the loss and accuracy for the training and testing data
    """
    history = results.history
    plt.figure(figsize=(8,4))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(oname + '_train.png', dpi=300)
    
def exe_load(args, df):
    """ Use a pretrained model
    """
    param, model = load_pretrained(args.model)
    df, close_scaler, scaler = preprocess(df)
    n_features = df.shape[1]
    n_per_in, n_per_out, _ = param
    predict(df, n_per_in, n_per_out, n_features, model, close_scaler, args.oname)
    

def exe_new(args, df):
    """ Build a new neural agent
    """
    nn = NN()
    nn.df, nn.close_scalar, nn.scalar = preprocess(df)
    nn.n_features = nn.df.shape[1]
    x, y = nn.split_sequence(nn.df.to_numpy())
    if args.model == 'lstm':
        # Using Simple LSTM Model
        nn.build_lstm()
    elif args.model == 'drnn':
        # Using DRNN Model (Stacked LSTM Arc)
        nn.build_drnn()
    else:
        raise ValueError('Model cannot found.')
    result = nn.model.fit(x, y, epochs=50, batch_size=128, validation_split=0.1, verbose=1)
    visualize_training_results(result, args.oname)
    predict(df, nn.n_per_in, nn.n_per_out, nn.n_features, nn.model, nn.close_scalar, args.oname)
    if args.save_model == 1:
        nn.model.save(args.oname)
        
def exe_q_l(args):
    name_file_data = args.path
    env_name = 'StockEnv-v0'

    num_episodes = 1 # training epoches
    render_p = 10 # print frequency
    init_balance = 1000 # initial fund agent have

    len_obs = 70 # observation length, number of days the agent look back
    len_window = 100 # num of times trained each time
    interval = 1 # interval of validation
    overlap = 20 # overlapping between two consecutive training data 
    batch_size = 1000 
    a = [i/10 for i in range(-4,5,1)]
    print(tuple(a))
    action_list=tuple(a)
    seed = 40
    
    train_data, test_data = get_data(f'../data/{name_file_data}')
    
    # Create an instant
    env = gym.make(env_name, train_data=train_data, eval_data=test_data, 
                   len_obs=len_obs, len_window=len_window, init_balance=init_balance,
                   action_list=action_list)
    env.seed(seed)
    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')
    # Create an agent
    agent = QNAgent(env.action_space.n, env.observation_space.shape, 
                    discount_rate=0.5, learning_rate=0.01, epsilon=0.01)
    init_ep = 0
    train_statistics = pd.DataFrame()
    test_statistics = pd.DataFrame()
    worth = []
    losses = []
    
    for ep in range(init_ep, num_episodes):
        _, _, loss = get_performance(env, agent, train_data=True, training=True, batch_size=batch_size)
        if (ep % render_p) == 0:
            env.render(ep)
        worth.append(env.net_worth)
        losses = np.concatenate((losses,loss))
        if ep % interval == 0:
            overlap = overlap
            results_train = np.empty(shape=(0, 3))
            results_test = np.empty(shape=(0, 3))

            size_test = ((len(env.eval_data)-env.len_obs-env.len_window) // overlap)+1
            cagr_train, vol_train, _ = get_performance(env, agent, train_data=True, training=False, batch_size=size_test)
            results_train = np.array([np.tile(ep, size_test), cagr_train, vol_train]).transpose()

            cagr_test, vol_test, _ = get_performance(env, agent, train_data=False, training=False, overlap=overlap, batch_size=size_test)
            results_test = np.array([np.tile(ep, size_test), cagr_test, vol_test]).transpose()

            train_statistics = pd.concat([train_statistics, pd.DataFrame(results_train, columns=['epoch', 'cagr','volatility'])])
            test_statistics = pd.concat([test_statistics, pd.DataFrame(results_test, columns=['epoch', 'cagr','volatility'])])
    
    precision = num_episodes
    threshold = 5000
    worth_mean = np.mean(worth,axis=-1)
    k = np.split(worth_mean, precision)
    k = np.mean(k, axis = -1)
    #k = [len(np.where(i > threshold)[0]) for i in worth]
    plt.figure(figsize=(8,4))
    plt.plot(k,'.-')
    plt.savefig(oname + '_reward.png', dpi=300)

def load_ql_agent(filename):
    # Obtain parameter
    param = pickle.load(open(filename + '/param', 'rb'))
    # Initialize Agent
    a = QNAgent(param[0], param[1])
    # Set parameter
    a.set_param(param)
    # Load model weight
    a.model.load_weights(filename+'/model')
    return a


if __name__ == '__main__':
    # Obtain argments from command line
    parser = argparse.ArgumentParser(description='Stock Prediction')
    parser.add_argument('-path', default='./data/AAPL.csv', type=str)
    parser.add_argument('-model', type=str, default='lstm')
    parser.add_argument('-load', type=int, default=1)
    parser.add_argument('-oname', type=str, default='result')
    parser.add_argument('-save_model', type=int, default=0)

    args = parser.parse_args()

    data = pd.read_csv(args.path, date_parser=True)
    if args.load == 1:
        exe_load(args, data)
    else:
        exe_new(args, data)
    
    agent = load_ql_agent('ql_model')
    print(agent.state_size)
    state = np.zeros(agent.state_size) # shape = (1,n)
    print(agent.get_action(np.transpose(state)))
    
    
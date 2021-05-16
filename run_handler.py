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

import math
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


def exe_q_l(args):
    pred_res = args.oname + '_pred.csv'
    env_name = 'StockEnv-v0'
    
    # Parameters
    init_balance = 1000 # initial fund agent have
    len_obs = 70 # observation length, number of days the agent look back
    a = [i/10 for i in range(-4,5,1)]
    action_list=tuple(a)
    
    # Data Handler
    hist_data = pd.read_csv(args.path, index_col=0, parse_dates=True, header=0)
    hist_data = hist_data.tail(len_obs+1)
    hist_data = hist_data[['Close']]
    pred_data = pd.read_csv(pred_res, index_col=0, parse_dates=True, header=0)
    data = pd.concat([hist_data, pred_data])
    data = pd.concat([data, data.pct_change()], axis=1).iloc[1:]
    data.columns = ['prices', 'returns']
    
    print(f'Action space: {action_list}')
    # Simulation
    agent = load_ql_agent()
    worth = []
    idx = np.arange(0,len(pred_data),1)
    prices = np.array([data.values[i:i+len_obs, 0] for i in idx])
    returns = np.array([data.values[i:i+len_obs, 1] for i in idx])
    balance = init_balance
    shares_held = 0
    for i in range(len(returns)):
        state = returns[i].reshape(1,returns.shape[1])
        act = agent.get_action(state, use_random=False)[0]
        current_price = prices[i][-1]
        mid = len(action_list) // 2
        if act > mid:
            # Buy 
            percentage = action_list[act]
            shares_bought = math.floor((balance / current_price) * percentage)
            balance -= shares_bought * current_price
            shares_held += shares_bought
        elif act == mid:
            pass
        elif act < mid:
            # Sell
            percentage = np.abs(action_list[act])
            try:
                shares_sold = math.floor(shares_held * percentage)
            except:
                print(shares_held)
            balance += shares_sold * current_price
            shares_held -= shares_sold 
        net_worth = balance + (shares_held * current_price)
        worth.append(net_worth)
                
    plt.figure(figsize=(8,4))
    plt.plot(worth,'.-')
    plt.savefig(args.oname + '_reward.png', dpi=300)

def load_ql_agent():
    # Obtain parameter
    param = pickle.load(open('pretrained_models/qlearn/param', 'rb'))
    # Initialize Agent
    a = QNAgent(param[0], param[1])
    # Set parameter
    a.set_param(param)
    # Load model weight
    a.model.load_weights('pretrained_models/qlearn/model')
    return a

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

    
    
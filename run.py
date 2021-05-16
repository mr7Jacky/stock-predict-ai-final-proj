import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
from datetime import timedelta
import ta
import keras 
import argparse

import sys
sys.path.insert(1, './price_pred')
from nn import NN

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
    nn.model.save(args.oname)
    


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
    
    
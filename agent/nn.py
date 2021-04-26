# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
# Technical Analysis library
import ta
# Neural Network library
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

plt.style.use("bmh")


class NN:

    def __init__(self):
        """
        Initialization
        """
        # Scale fitting the close prices separately for inverse_transformations purposes later
        self.close_scalar = RobustScaler()
        # Normalizing/Scaling the DF
        self.scalar = RobustScaler()
        self.df = None
        # How many periods looking back to learn
        self.n_per_in = 90
        # How many periods to predict
        self.n_per_out = 30
        # Features
        self.n_features = 0
        # Instantiating the model
        self.model = Sequential()

    def preprocess(self, data):
        """
        Preprocess the data
        :param data: the data frame to process
        """
        # Datetime conversion
        data['Date'] = pd.to_datetime(data.Date)
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
        self.close_scalar.fit(data[['Close']])
        data = pd.DataFrame(self.scalar.fit_transform(data), columns=data.columns, index=data.index)
        self.df = data
        self.n_features = self.df.shape[1]

    def split_sequence(self, seq):
        """
        Splits the multivariate time sequence
        """
        # Creating a list for both variables
        x, y = [], []
        for i in range(len(seq)):
            # Finding the end of the current sequence
            end = i + self.n_per_in
            out_end = end + self.n_per_out
            # Breaking out of the loop if we have exceeded the dataset's length
            if out_end > len(seq):
                break
            # Splitting the sequences into: x = past prices and indicators, y = prices ahead
            seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)

    @staticmethod
    def visualize_training_results(results):
        """
        Plots the loss and accuracy for the training and testing data
        """
        history = results.history
        plt.figure(figsize=(16, 5))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        plt.figure(figsize=(16, 5))
        plt.plot(history['val_accuracy'])
        plt.plot(history['accuracy'])
        plt.legend(['val_accuracy', 'accuracy'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

    def layer_maker(self, n_layers, n_nodes, activation, drop=None, d_rate=.5):
        """
        Creates a specified number of hidden layers for an RNN
        Optional: Adds regularization option - the dropout layer to prevent potential over-fitting (if necessary)
        """
        # Creating the specified number of hidden layers with the specified number of nodes
        for x in range(1, n_layers + 1):
            self.model.add(LSTM(n_nodes, activation=activation, return_sequences=True))
            # Adds a Dropout layer after every Nth hidden layer (the 'drop' variable)
            try:
                if x % drop == 0:
                    self.model.add(Dropout(d_rate))
            except:
                pass

    def build_nn(self):
        """
        Create neural network
        """
        # Activation
        activation = "tanh"
        # Input layer
        self.model.add(LSTM(90,
                            activation=activation,
                            return_sequences=True,
                            input_shape=(self.n_per_in, self.n_features)))
        # Hidden layers
        self.layer_maker(n_layers=1, n_nodes=30, activation=activation)
        # Final Hidden layer
        self.model.add(LSTM(60, activation=activation))
        # Output layer
        self.model.add(Dense(self.n_per_out))
        # Model summary
        self.model.summary()
        # Compiling the data with selected specifications
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    def train_nn(self, train_x, train_y):
        """
        Train neural network by giving training sets
        :return: the training result
        """
        # Fitting and Training
        result = self.model.fit(train_x, train_y, epochs=50, batch_size=128, validation_split=0.1)
        return result

    def validater(self):
        """
        Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
        Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
        """
        # Creating an empty DF to store the predictions
        predictions = pd.DataFrame(index=self.df.index, columns=[self.df.columns[0]])
        for i in range(self.n_per_in, len(self.df) - self.n_per_in, self.n_per_out):
            # Creating rolling intervals to predict off of
            x = self.df[-i - self.n_per_in:-i]
            # Predicting using rolling intervals
            y_hat = self.model.predict(np.array(x).reshape(1, self.n_per_in, self.n_features))
            # Transforming values back to their normal prices
            y_hat = self.close_scalar.inverse_transform(y_hat)[0]
            # DF to store the values and append later, frequency uses business days
            pred_df = pd.DataFrame(y_hat, index=pd.date_range(start=x.index[-1], periods=len(y_hat), freq="B"),
                                   columns=[x.columns[0]])
            # Updating the predictions DF
            predictions.update(pred_df)
        return predictions

    @staticmethod
    def val_rmse(df1, df2):
        """
        Calculates the root mean square error between the two Dataframes
        """
        df = df1.copy()
        # Adding a new column with the closing prices from the second DF
        df['close2'] = df2.Close
        # Dropping the NaN values
        df.dropna(inplace=True)
        # Adding another column containing the difference between the two DFs' closing prices
        df['diff'] = df.Close - df.close2
        # Squaring the difference and getting the mean
        rms = (df[['diff']] ** 2).mean()
        # Returning the square root of the root mean square
        return float(np.sqrt(rms))

    def prediction_vs_actual(self):
        """
        Compare the future price and actual price
        """
        # Transforming the actual values to their original price
        actual = pd.DataFrame(self.close_scalar.inverse_transform(self.df[["Close"]]),
                              index=self.df.index,
                              columns=[self.df.columns[0]])
        # Getting a DF of the predicted values to validate against
        predictions = self.validater()
        # Printing the RMSE
        print("RMSE:", self.val_rmse(actual, predictions))
        # Plotting
        plt.figure(figsize=(16, 6))
        # Plotting those predictions
        plt.plot(predictions, label='Predicted')
        # Plotting the actual values
        plt.plot(actual, label='Actual')
        plt.title(f"Predicted vs Actual Closing Prices")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def forecast_future(self):
        """
        Forecast the future prices
        """
        # Predicting off of the most recent days from the original DF
        y_hat = self.model.predict(np.array(self.df.tail(self.n_per_in)).reshape(1, self.n_per_in, self.n_features))
        # Transforming the predicted values back to their original format
        y_hat = self.close_scalar.inverse_transform(y_hat)[0]
        # Creating a DF of the predicted prices
        preds = pd.DataFrame(y_hat, index=pd.date_range(start=self.df.index[-1] + timedelta(days=1),
                                                        periods=len(y_hat), freq="B"),
                             columns=[self.df.columns[0]])
        # Number of periods back to plot the actual values
        pers = self.n_per_in
        # Transforming the actual values to their original price
        actual = pd.DataFrame(self.close_scalar.inverse_transform(self.df[["Close"]].tail(pers)),
                              index=self.df.Close.tail(pers).index,
                              columns=[self.df.columns[0]]).append(preds.head(1))
        # Printing the predicted prices
        print(preds)
        # Plotting
        plt.figure(figsize=(16, 6))
        plt.plot(actual, label="Actual Prices")
        plt.plot(preds, label="Predicted Prices")
        plt.ylabel("Price")
        plt.xlabel("Dates")
        plt.title(f"Forecasting the next {len(y_hat)} days")
        plt.legend()
        plt.show()


# # Loading in the Data
# import os, sys
# sys.path.append("../stock_env/")
# from account import Account
# acc = Account("../stock_env/stock_info/")
nn = NN()
df = pd.read_csv("AAPL.csv")
nn.preprocess(df)
# Splitting the data into appropriate sequences
x, y = nn.split_sequence(nn.df.to_numpy())
res = nn.build_nn(x, y)
nn.prediction_vs_actual()
nn.forecast_future()
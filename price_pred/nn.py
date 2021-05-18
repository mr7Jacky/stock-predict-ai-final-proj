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

    def __init__(self, n_per_in=90, n_per_out=30):
        """
        Initialization
        :param n_per_in: Input shape of neural network
        :param n_per_out: output shape of neural network
        """
        # Scale fitting the close prices separately for inverse_transformations purposes later
        self.close_scalar = RobustScaler()
        # Normalizing/Scaling the DF
        self.scalar = RobustScaler()
        self.df = None
        # How many periods looking back to learn
        self.n_per_in = n_per_in
        # How many periods to predict
        self.n_per_out = n_per_out
        # Features (This value would be updated in preprocess step)
        self.n_features = 0
        # Instantiating the model
        self.model = None

    def preprocess(self, data):
        """
        Preprocess the data and use robust scaler to fit the data
        :param data: the data frame to process
        """
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
        self.close_scalar.fit(data[['Close']])
        data = pd.DataFrame(self.scalar.fit_transform(data), columns=data.columns, index=data.index)
        self.df = data
        self.n_features = self.df.shape[1]

    def split_sequence(self, seq):
        """
        Splits the multivariate time sequence
        :param seq: sequence to split
        :return: two split sequences
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
        :param results: the training results to plot
        """

        history = results.history
        plt.figure(figsize=(8, 4))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def build_lstm(self):
        """
        Create a simple one-lstm layer for neural network
        """
        # Activation
        self.model = Sequential()
        activation = "tanh"
        # Input layer
        self.model.add(LSTM(90,
                            activation=activation,
                            return_sequences=True,
                            input_shape=(self.n_per_in, self.n_features)))
        # Hidden layers
        self.model.add(Dropout(0.5))
        # Final Hidden layer
        self.model.add(LSTM(60, activation=activation))
        # Output layer
        self.model.add(Dense(self.n_per_out))
        # Model summary
        self.model.summary()
        # Compiling the data with selected specifications
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    def build_drnn(self):
        """
        Create a deep recurrent neural network based on stacked lstm architecture
        """
        # Activation
        self.model = Sequential()
        activation = "tanh"
        # Input layer
        self.model.add(LSTM(128, activation=activation,
                            return_sequences=True,
                            input_shape=(self.n_per_in, self.n_features)))
        # Hidden layers
        self.model.add(LSTM(256, activation=activation, return_sequences=True))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(512, activation=activation, return_sequences=True))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(256, activation=activation, return_sequences=True))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(128, activation=activation, return_sequences=True))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(64, activation=activation, return_sequences=True))
        self.model.add(Dropout(0.25))
        # Final Hidden layer
        self.model.add(LSTM(32, activation=activation))
        # Output layer
        self.model.add(Dense(self.n_per_out))
        # Model summary
        self.model.summary()
        # Compiling the data with selected specifications
        self.model.compile(optimizer='adam', loss='huber')

    def train_nn(self, train_x, train_y):
        """
        Train neural network by giving training sets
        :param train_x: training input values
        :param train_y: training output targets
        :return: the training result
        """
        # Fitting and Training
        result = self.model.fit(train_x, train_y, epochs=50, batch_size=128, validation_split=0.1)
        return result

    def validater_at_date(self, date=None):
        """
        Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
        :param date: date to predict
        :return a DF containing the predicted values for the model with the corresponding index values based on a
                business day frequency
        """
        if date is None:
            date = self.n_per_in
        elif date < self.n_per_in:
            raise ValueError
        # Creating an empty DF to store the predictions
        predictions = pd.DataFrame(index=self.df.index, columns=[self.df.columns[0]])
        for i in range(date, len(self.df) - date, self.n_per_out):
            # Creating rolling intervals to predict off of
            x = self.df[-i - date:-i]
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
        :param df1: the first dataframe to compare
        :param df2: the second dataframe to compare
        :return: the square root of the root mean square
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
        Compare the future price and actual price by plotting the predicted results and actual prices together
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
        plt.figure(figsize=(8, 4))
        # Plotting those predictions
        plt.plot(predictions, label='Predicted')
        # Plotting the actual values
        plt.plot(actual, label='Actual')
        plt.title(f"Predicted vs Actual Closing Prices")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    # Modify to include other dfs
    def validater(self):
        """
        Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
        :returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
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

    def forecast_future(self):
        """
        Forecast the future prices by plotting the predicted results
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
        # Plotting
        plt.figure(figsize=(8, 4))
        plt.plot(actual, label="Actual Prices")
        plt.plot(preds, label="Predicted Prices")
        plt.ylabel("Price")
        plt.xlabel("Dates")
        plt.title(f"Forecasting the next {len(y_hat)} days")
        plt.legend()
        plt.show()


def prediction()
    """
    Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
    Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
    """
    # Load model from pretrain model
    
    # Process the data
    
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
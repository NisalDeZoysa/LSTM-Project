import numpy as np
import pandas as pd
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import joblib

start = '2013-02-08'
end = '2018-02-07'

st.title('Stock market Prediction of TESLA')

user_input = 'AAPL'

yf.pdr_override()
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2013 to 2018')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig,'b')

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df['Close'].rolling(window=100).mean()
ma200 = df['Close'].rolling(window=200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


#train_data = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
#test_data = pd.DataFrame(df['Close'][0:int(len(df)*0.7): int(len(df))])


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0,1))

#model=load_model('LSTM-Stock-Market.keras')

#past_209_data = data_training.tail(209)
#final_df = past_209_data.append(data_testing, ignore_index=True)

#x_test= []
#y_test= []




# Load model and scaler
model = load_model('LSTM-Stock-Market.keras')
scaler = joblib.load('stock-scaler.sav')

# Define start and end dates for historical data
start = '2013-02-08'
end = '2021-06-20'  # Adjust end date as needed

# Load stock data using Yahoo Finance API
user_input = 'AAPL'
yf.pdr_override()
df = yf.download(user_input, start=start, end=end)

# Ensure enough data for predictions (at least 50 days)
if len(df) < 50:
    st.error('Insufficient data to make predictions. Please choose a longer period.')
else:
    # Create rolling window of past 50 days data
    set_data = [df['Close'][i-50:i].values for i in range(50, len(df)+1)]

    # Prepare arrays to store predictions
    predictions = []

    # Iterate through each set of 50 days, make predictions, and store them
    for data_set in set_data:
        data_set_scaled = scaler.transform(data_set.reshape(1, -1))  # Ensure it's (1, 50)
        prediction_scaled = model.predict(np.expand_dims(data_set_scaled, axis=0))
        prediction = scaler.inverse_transform(prediction_scaled).flatten()[0]
        predictions.append(prediction)

    # Prepare actual data for plotting
    actual_dates = df.index[50:]  # Dates corresponding to predictions
    actual_prices = df['Close'][50:].values  # Actual closing prices

    # Plotting in Streamlit
    st.title('Stock Market Prediction of AAPL Stocks')
    st.subheader('Actual vs Predicted Closing Prices')

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_dates, actual_prices, 'b', label='Actual')
    ax.plot(actual_dates, predictions, 'g', label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.legend()

    # Display the plot using Streamlit's pyplot
    st.pyplot(fig)
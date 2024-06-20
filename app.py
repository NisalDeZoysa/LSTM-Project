import numpy as np
import pandas as pd
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf

start = '2013-02-08'
end = '2018-02-07'

st.title('Stock market Prediction of TESLA stocks')

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




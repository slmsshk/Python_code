import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from indicators import *
import pandas_ta as ta
import pandas as pd
import time
import threading


# Define a function to fetch data
def fetch_data(pair_X, interval, period):
    ticker = yf.Ticker(pair_X)
    hist = ticker.history(interval=interval, period=period)
    return hist

# ======================================================== #
# Data prep
# ======================================================== #

majors = [
    'USDJPY=X', 'EURUSD=X', 'AUDUSD=X',
    'GBPUSD=X', 'NZDUSD=X', 'USDCAD=X', 'USDCHF=X'
]

period_options = {
    '1m': ['1d', '5d', '1wk'],
    '2m': ['1d', '5d', '1wk', '1mo'],
    '5m': ['1d', '5d', '1wk', '1mo'],
    '15m': ['1d', '5d', '1wk', '1mo'],
    '30m': ['1d', '5d', '1wk', '1mo'],
    '60m': ['1d', '5d', '1wk', '1mo', '2mo', '3mo', '6mo', '1y', '2y', 'ytd'],
    '1d': ['1d', '5d', '1wk', '1mo', '2mo', '3mo', '6mo', '1y', '3y', '5y', '10y', 'ytd', 'max'],
    '5d': ['1mo', '3mo', '6mo', '1y', '3y', '5y', '10y', 'max'],
    '1wk': ['1mo', '3mo', '6mo', '1y', '3y', '5y', '10y', 'max'],
    '1mo': ['3mo', '6mo', '1y', '3y', '5y', '10y', 'max'],
    '3mo': ['1y', '3y', '5y', '10y', 'max'],
}
st.sidebar.write(f"<p style='font-size:15px;background-color:	#5d8aa8;color:	#f2f3f4;'>Parameters to fetch Data & Train models</p>",unsafe_allow_html=True)

# We can loop over each currency pair, interval, and period to create and display the charts
majors=['USDJPY','EURUSD','AUDUSD','GBPUSD','NZDUSD','USDCAD','USDCHF']
pair=st.sidebar.selectbox(label='Select the pair you want to trade',options=majors)


interval=st.sidebar.selectbox(label='Select Interval',options=('1m', '2m', '5m', '15m', '30m', '60m', '1d', '5d', '1wk', '1mo', '3mo'))

period='1d'
# # 1m,
if interval =="1m":
    periods=st.sidebar.selectbox(label='Enter Period',options=['1d','5d','1wk'])
# # 2m,5m,15m,30m
elif interval in ["2m",'5m', '15m','30m']:
    period=st.sidebar.selectbox(label='Select Period',options=['1d','5d','1wk','1mo'])
# # 1hr
elif interval in ['60m']:
    period=st.sidebar.selectbox(label='Select Period',options=['1d','5d','1wk','1mo','2mo','3mo','6mo','1y','2y','ytd'])
    # periods=st.selectbox(label='Enter Period',options=['1d','5d','1wk','1mo',])
else:
    period=st.sidebar.selectbox(label='Select Period',options=['1d','5d','1wk','1mo','2mo','3mo','6mo','1y','3y','5y','10y','ytd','max'])
ma_window = st.sidebar.number_input('Enter the MA term', min_value=1, max_value=15, value=5, step=1)

# eur=yf.Ticker(pair_X)

pair_X = pair + "=X"



# Create an empty placeholder for data
data_placeholder = st.empty()

fig = go.Figure()

# Function to update the chart
def update_chart():
    while True:
        # Fetch data periodically (every 10 seconds)
        hist = fetch_data(pair_X, interval, period)

        # Clear the existing data on the figure
        fig.data = []

        # Add the new data to the figure
        fig.add_trace(go.Candlestick(x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name=pair_X))

        # Update the placeholder with the latest Plotly figure
        with data_placeholder:
            st.plotly_chart(fig, use_container_width=True)

        # Pause for 10 seconds before fetching data again
        time.sleep(10)

# Create a thread for updating the chart
chart_update_thread = threading.Thread(target=update_chart)

# Start the thread
chart_update_thread.start()
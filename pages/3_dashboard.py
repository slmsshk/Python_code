import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# Function to fetch stock data
def fetch_data(ticker):
    today = datetime.today().strftime('%Y-%m-%d')
    past_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=past_date, end=today)
    return data['Close']

# Initialize user portfolio
portfolio = {
    'cash': 10000.0, # Starting cash
    'stocks': {} # Dictionary to store owned stocks and quantities
}

# Function to buy stocks
def buy_stock(ticker, amount, price):
    cost = amount * price
    if cost <= portfolio['cash']:
        portfolio['cash'] -= cost
        if ticker in portfolio['stocks']:
            portfolio['stocks'][ticker] += amount
        else:
            portfolio['stocks'][ticker] = amount
        return True
    else:
        return False

# Function to sell stocks
def sell_stock(ticker, amount, price):
    if ticker in portfolio['stocks'] and portfolio['stocks'][ticker] >= amount:
        portfolio['cash'] += amount * price
        portfolio['stocks'][ticker] -= amount
        return True
    else:
        return False
st.title('Stock Market Simulation Game')

# Sidebar for user input
st.sidebar.header('Trading Panel')
ticker = st.sidebar.text_input('Enter a stock symbol', value='AAPL').upper()
amount = st.sidebar.number_input('Enter amount', min_value=1, value=1)
buy = st.sidebar.button('Buy')
sell = st.sidebar.button('Sell')

# Fetch and display stock data
if ticker:
    data = fetch_data(ticker)
    st.line_chart(data)

    # Show current cash and stocks
    st.write(f"Cash: ${portfolio['cash']}")
    st.write('Stocks:', portfolio['stocks'])

    # Buying stocks
    if buy:
        if buy_stock(ticker, amount, data[-1]):  # Assuming the latest price for simplicity
            st.success(f"Bought {amount} shares of {ticker}")
        else:
            st.error("Not enough cash to buy")

    # Selling stocks
    if sell:
        if sell_stock(ticker, amount, data[-1]):  # Assuming the latest price for simplicity
            st.success(f"Sold {amount} shares of {ticker}")
        else:
            st.error("Not enough shares to sell")

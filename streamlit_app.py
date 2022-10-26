import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import time

# configure page
st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="❣",
    layout="wide",
)


def header(url):
     st.markdown(f'<p style="background-color:#f2f3f4;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)


header('Stock Price')

# while True:
# for i in range(10):
tick=yf.Ticker("EURUSD=X")

data=tick.history(interval='1m',period='max')

st.write(data.head())

line_chart=px.line(data_frame=data,y='Close')
st.write(line_chart)
# time.sleep(1)
print('Newiterationd')
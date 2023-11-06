import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


majors=['USDJPY','EURUSD','AUDUSD','EURUSD','GBPUSD','NZDUSD','USDCAD','USDCHF']

pair=st.sidebar.selectbox(label='Select the pair you want to trade',options=majors)+"=X"

st.title(pair[:-2]+"!!!")


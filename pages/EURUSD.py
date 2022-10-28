import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Eur/USD", page_icon='pages\eurusd.png')

# def icon(url):
    


st.title('EUR/USD;')
eur=yf.Ticker("EURUSD=X")
hist = eur.history(interval='1h',period='s')
fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False)

st.write(fig)
 
fig=px.line(data_frame=hist,y='Close')

st.write(fig)

st.write(f"<p style='font-size:15px;'>Preparing Data</p>")
X=[]
Y=[]
end=3
x=hist.Close.dropna().values


for i in range(len(x)):
    end+=1
    if end>=len(x)-1:break
    X.append(x[i:end])
    Y.append(x[end])

from keras.models import Sequential
from keras.layers import LSTM,Dense



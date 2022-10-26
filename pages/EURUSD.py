import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

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

# fig = px.line(hist,y='Close').Figure(data=[go.Candlestick(x=hist.index,
#                 open=hist['Open'],
#                 high=hist['High'],
#                 low=hist['Low'],
#                 close=hist['Close'])])
# fig.update_layout(xaxis_rangeslider_visible=False)

st.write(fig)

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

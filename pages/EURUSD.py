from functools import cache
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


st.set_page_config(page_title="Eur/USD", page_icon='eurusd.png',initial_sidebar_state = "expanded")

st.title('EUR/USD!!!!')

st.markdown(f"<p style='background-color:darkgreen;padding:5px;color:#f2f3f4;'>Fetching Data</p>",unsafe_allow_html=True)

# =================================================================
# Input function for interval and period
# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo] 
#  Valid periods: 1d,5d,1wk,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

st.sidebar.write(f"<p style='font-size:15px;background-color:	#5d8aa8;color:	#f2f3f4;'>Parameters to fetch Data</p> <p style='font-size:10px;'><em>enter interval and period to get rid of the error<em></p>",unsafe_allow_html=True)
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

st.code(f"""
import yfinance as yf
eur=yf.Ticker("EURUSD=X")
hist = eur.history(interval={interval},period={period})
""")

eur=yf.Ticker("EURUSD=X")
hist = eur.history(interval=interval,period=period)
eur.history()

# ==================================================================
# Candle Stick Chart
# st.write('Candle Stick for Close Column')
fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False)

st.write(fig)
 
fig=px.line(data_frame=hist,y='Close')

st.write(fig)
# ===================================================================


st.write(f"<p style='font-size:15px;'>Preparing Data</p>",unsafe_allow_html=True)

st.sidebar.write(f"<p style='font-size:15px;background-color:	#5d8aa8;color:	#f2f3f4;'>Parameters for model training</p>",unsafe_allow_html=True)

end=st.sidebar.number_input("Enter how many lags you want to use",value=1,min_value=1,max_value=10,step=1)
sp=end
future=st.sidebar.number_input("Enter how many steps you want to predict in future",value=1,min_value=1,max_value=3,step=1)

st.code("""
X=[]
Y=[]
x=hist.Close.dropna().values

for i in range(len(x)):
    end+=1
    if end>=len(x)-1:break
    X.append(x[i:end])
    Y.append(x[end])""")

X=[]
Y=[]
x=hist.Close.dropna().values

for i in range(len(x)):
    if end>=len(x)-1:break
    X.append(x[i:end])
    Y.append(x[end:end+future])
    end+=1

col1,col2=st.columns(2)
col1.write((X.pop(-1),Y.pop(-1)))
col2.write((X[-5:],Y[-5:]))

import numpy as np

X_arr,Y_arr=np.array(X),np.array(Y)
X_arr=X_arr.reshape(X_arr.shape[:][0],sp,1)

X_train,X_test,y_train,y_test=X_arr[:-100],X_arr[-100:],Y_arr[:-100],Y_arr[-100:]

st.write(f'input data shape{X_train.shape}')

st.sidebar.image("pages/images.jpg")
# ================================================================
# Model Training
col3,col4,col5=st.columns(3)
col4.header('Model Training')

from keras.models import Sequential
from keras.layers import LSTM,Dense

#Architecture
nn=Sequential(name='Sequence_LSTM')
nn.add(LSTM(50,activation='relu',input_shape=(sp,1),name='input_layer_lstm'))
nn.add(Dense(50,name='Hidden_layer_Dense1',activation='LeakyReLU'))
nn.add(Dense(50,name='Hidden_layer_Dense2'))
nn.add(Dense(future,name='Output_layer_Dense'))
nn.compile(loss='mse',optimizer='adam')
nn.summary(print_fn=lambda x: st.text(x))

# print(nn.summary())
# st.write(f'<p>{nn.summary()}</p>',unsafe_allow_html=True)
nn.fit(X_train,y_train,epochs=100,batch_size=100)


pred=nn.predict(X_test)

# plt.plot(hist.Close)
fig,ax=plt.subplots()
ax.plot(pred,label='prediction',color='orange')
ax.plot(y_test,label='Actual',color='red')
# plt.xticks(hist.index[:-100])
ax.legend()
st.pyplot(fig)

fp = eur.history(interval='1m',period='60m')
fp.Close.values

# X=[]
# # end=3
# x=fp.Close.values
# for i in range(len(x)):
#     end+=1
#     if end>=len(x)-1:break
#     X.append(x[i:end])
# #     Y.append(x[end])

# # import numpy as np
# new_points=np.array(X).reshape(np.array(X).shape[0], 4,1)
# new_points.shape

# pred1=nn.predict(new_points)

# fig1,ax1=plt.subplots()
# ax1.plot(x,label='real')
# ax1.plot(pred1)
# ax1.legend()
# st.pyplot(fig1)
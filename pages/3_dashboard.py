import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from indicators import *
import pandas_ta as ta
import pandas as pd
from prophet import Prophet

st.title('Forex Dashboard')

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

# We can loop over each currency pair, interval, and period to create and display the charts
majors=['USDJPY','EURUSD','AUDUSD','EURUSD','GBPUSD','NZDUSD','USDCAD','USDCHF']

st.sidebar.write(f"<p style='font-size:15px;background-color:	#5d8aa8;color:	#f2f3f4;'>Parameters to fetch Data</p>",unsafe_allow_html=True)

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

pair=st.sidebar.selectbox(label='Select the pair you want to trade',options=majors)
# eur=yf.Ticker(pair_X)

pair_X = pair + "=X"
# p = st.sidebar.number_input(min_value=1,label='Enter the  MA term',max_value=15,step=1)

ticker=yf.Ticker(pair_X)
hist = ticker.history(interval=interval,period=period)
# hist = eur.history(interval=interval,period=period)
# eur.history()

# Fetch historical data
ticker = yf.Ticker(pair_X)
hist = ticker.history(interval=interval, period=period)

# Calculate indicators
# Moving Average
hist['MA'] = hist['Close'].rolling(window=ma_window).mean()

# MACD
hist['MACD'] = ta.macd(hist['Close'])['MACD_12_26_9']
hist['MACDSignal'] = ta.macd(hist['Close'])['MACDs_12_26_9']

# RSI
hist['RSI'] = ta.rsi(hist['Close'])

# Bollinger Bands
bollinger = ta.bbands(hist['Close'])
st.write(bollinger.columns,bollinger)
hist['Bollinger_Lower'] = bollinger['BBL_5_2.0']
hist['Bollinger_Middle'] = bollinger['BBM_5_2.0']  # Middle Band might not be used, but included for reference
hist['Bollinger_Upper'] = bollinger['BBU_5_2.0']

# ADX
adx_indicator = ta.adx(hist['High'], hist['Low'], hist['Close'])
hist['ADX'] = adx_indicator['ADX_14']

# Parabolic SAR
hist['ParabolicSAR'] = calculate_parabolic_sar(hist['High'], hist['Low'])


# Ichimoku Cloud
# Note: Ichimoku Cloud returns several values, so select the ones you want to use for trend analysis.
# ichimoku = ta.ichimoku(hist['High'], hist['Low'], hist['Close'])
# ichimoku_df = ichimoku[0]
# # Now you can access the components with string indices
# hist['Ichimoku_Base'] = ichimoku_df['ISA_9']  # For the Base line
# hist['Ichimoku_Conversion'] = ichimoku_df['ISB_26']  # For the Conversion line
# # And so on for other components like 'ISA_26', 'ISB_52', 'ITS_9', 'IKS_26', 'ICS_26'

# If you want to add Leading Span A and B which could be in the second DataFrame of the tuple
# ichimoku_span_df = ichimoku[1]  # This is only necessary if the spans are in a separate DataFrame.
# ichimoku_span_df
# hist['Ichimoku_Leading_Span_A'] = ichimoku_span_df['ISA_9']
# hist['Ichimoku_Leading_Span_B'] = ichimoku_span_df['ISB_26']



# hist['Bollinger_Upper'] = bollinger['BBU_20_2.0']
# hist['Bollinger_Lower'] = bollinger['BBL_20_2.0']

# Prepare plotly figure
fig = go.Figure()

# Add OHLC or candlestick chart
fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name='Candlesticks'
))

# Add MA
fig.add_trace(go.Scatter(
    x=hist.index,
    y=hist['MA'],
    line=dict(color='blue', width=2),
    name='MA'
))
hist
# Interpret indicators for the trend
trend_analysis = {}

# Moving Average Trend Analysis
trend_analysis['MA Trend'] = 'Bullish' if hist['Close'].iloc[-1] > hist['MA'].iloc[-1] else 'Bearish'
# ADX Trend Analysis
adx_trend = 'Strong Trend' if hist['ADX'].iloc[-1] > 25 else ('Weak Trend' if hist['ADX'].iloc[-1] < 20 else 'Neutral')

# Parabolic SAR Trend Analysis
parabolic_sar_trend = 'Bullish' if hist['Close'].iloc[-1] > hist['ParabolicSAR'].iloc[-1] else 'Bearish'

trend_analysis['ADX Trend'] = adx_trend
trend_analysis['Parabolic SAR Trend'] = parabolic_sar_trend
# MACD Trend Analysis
trend_analysis['MACD Trend'] = 'Bullish' if hist['MACD'].iloc[-1] > hist['MACDSignal'].iloc[-1] else 'Bearish'

# RSI Trend Analysis
trend_analysis['RSI Trend'] = 'Bullish' if hist['RSI'].iloc[-1] < 30 else ('Bearish' if hist['RSI'].iloc[-1] > 70 else 'Neutral')

# Bollinger Bands Trend Analysis
trend_analysis['Bollinger Trend'] = 'Bullish' if hist['Close'].iloc[-1] < hist['Bollinger_Lower'].iloc[-1] else ('Bearish' if hist['Close'].iloc[-1] > hist['Bollinger_Upper'].iloc[-1] else 'Neutral')

# trend_analysis['Ichimoku_Trend'] = hist.apply(lambda row: 'Bullish' if row['Close'] > row['Ichimoku_Leading_Span_A'] and row['Close'] > row['Ichimoku_Leading_Span_B'] else 'Bearish', axis=1)
# Determine the Ichimoku trend
# if hist['Close'].iloc[-1] > hist['Ichimoku_Leading_Span_A'].iloc[-1] and hist['Close'].iloc[-1] > hist['Ichimoku_Leading_Span_B'].iloc[-1]:
#     trend_analysis['ichimoku_trend'] = 'Bullish'
# elif hist['Close'].iloc[-1] < hist['Ichimoku_Leading_Span_A'].iloc[-1] and hist['Close'].iloc[-1] < hist['Ichimoku_Leading_Span_B'].iloc[-1]:
#     trend_analysis['ichimoku_trend'] = 'Bearish'
# else:
#     trend_analysis['ichimoku_trend'] = 'Neutral'

# Add trend analysis to chart
for indicator, trend in trend_analysis.items():
    fig.add_annotation(x=hist.index[-1], y=hist['MA'].iloc[-1], text=f"{indicator}: {trend}", showarrow=False)

# Update layout
fig.update_layout(
    title=f'{pair} Analysis Chart',
    yaxis_title='Price',
    xaxis_title='Date',
    legend_title='Legend',
    xaxis_rangeslider_visible=False,
)

# Display chart
st.plotly_chart(fig, use_container_width=True)
# hist['Datetime'] = pd.to_datetime(hist.index)

# hist.set_index('Datetime', inplace=True)
close_prices = hist['Close']

# The auto_arima function will try to identify the most optimal parameters for the ARIMA model
# and return a fitted model.
import pmdarima as pm

model = pm.auto_arima(close_prices, seasonal=False, m=1,
                      d=None,   # let the model determine 'd'
                      start_p=1, start_q=1, 
                      max_p=3, max_q=3, 
                      trace=True, 
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

# Future value Variable
n_periods = st.sidebar.number_input('how many future steps',max_value=10,min_value=1,step=1)  # for example, predicting the next 5 periods

# Summarize the model

but = st.button('ARIMA SUMMARY')

if but:
    st.write(model.summary())
    observed_trace = go.Scatter(
        x=close_prices.index,
        y=close_prices.values,
        mode='lines',
        name='Observed'
    )


fc, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

# Generate a range of future dates equal to n_periods
future_index = pd.date_range(close_prices.index[-1], periods=2, freq='B')

if but:
    # Plot forecast values
    forecast_trace = go.Scatter(
        x=future_index,
        y=fc,
        mode='lines',
        name='Forecast'
    )

    # Plot confidence intervals
    confidence_band = go.Scatter(
        x=future_index.tolist() + future_index[::-1].tolist(),  # x, then x reversed
        y=conf_int[:, 1].tolist() + conf_int[::-1, 0].tolist(),  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    )

    # Create figure
    fig = go.Figure(data=[observed_trace, forecast_trace, confidence_band])

    # Update layout
    fig.update_layout(
        title='Forecast vs Observed',
        yaxis_title='Value',
        xaxis_title='Date'
    )

    # Show plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Prophet

df = pd.DataFrame({
    'ds': pd.to_datetime(hist.index),
    'y': hist['Close']
})
df['ds'] = df['ds'].dt.tz_localize(None)

m = Prophet()

# Fit the model with your dataframe
m.fit(df)

# Create a dataframe for future dates to predict
future = m.make_future_dataframe(periods=n_periods)  

# Predict the values for the future dates
forecast = m.predict(future)

pro = st.button('Prophet Viz',  )

if pro:
# Plot fig1 = m.plot(forecast)
    fig1 = m.plot(forecast)
    st.write('Forecast plot')
    st.pyplot(fig1)

    # # Plot the forecast components
    # fig2 = m.plot_components(forecast)
    # st.write('Forecast components')
    # st.pyplot(fig2)



# st.write(hist.head())
# Display trend analysis results
st.write(pd.DataFrame(trend_analysis,index=['Trend']))


models = pd.DataFrame()

models['Arima'] = ['Greator' if close_prices.iloc[-1]>=fc.iloc[0] else 'less']
models['Prophet'] = ['Greator' if forecast['yhat'][0] >=fc.iloc[0] else 'less']

st.write(models)


# Define a function to calculate the moving average and determine the trend
def calculate_moving_average(data, period):
    return data.rolling(window=period).mean()


def determine_trend(moving_average):
    if moving_average.iloc[-1] > moving_average.iloc[-2]:
        return "Bullish"
    elif moving_average.iloc[-1] < moving_average.iloc[-2]:
        return "Bearish"
    else:
        return "Neutral"
    
import pandas as pd


def calculate_parabolic_sar(high, low, acceleration_factor=0.02, max_acceleration_factor=0.2):
    # Initialize the SAR array with the first value of the high series
    sar = [high[0]]
    high_point = high[0]
    low_point = low[0]
    long = True  # Initial position
    af = acceleration_factor  # Initial acceleration factor

    for i in range(1, len(high)):
        if long:
            sar.append(max(sar[-1] + af * (high_point - sar[-1]), low[i-1], low[i-2 if i > 1 else i]))
            if high[i] > high_point:
                high_point = high[i]
                af = min(af + acceleration_factor, max_acceleration_factor)
            if low[i] < sar[-1]:
                long = False
                low_point = low[i]
                sar[-1] = high_point
                af = acceleration_factor
        else:
            sar.append(min(sar[-1] + af * (low_point - sar[-1]), high[i-1], high[i-2 if i > 1 else i]))
            if low[i] < low_point:
                low_point = low[i]
                af = min(af + acceleration_factor, max_acceleration_factor)
            if high[i] > sar[-1]:
                long = True
                high_point = high[i]
                sar[-1] = low_point
                af = acceleration_factor

    return pd.Series(sar, index=high.index)
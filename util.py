import datetime as dt
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime
import yfinance as yf
import re
import matplotlib.pyplot as plt
import os
import socket

def get_sharpe(return_stream):
    
    sharpe_ratio = (return_stream.mean() / return_stream.std()) * np.sqrt(252)
    
    return sharpe_ratio


def get_sortino(return_stream):
    
    downside_returns = return_stream[return_stream < 0]
    
    sortino_ratio = (return_stream.mean() / downside_returns.std()) * np.sqrt(252)
    
    return sortino_ratio


def get_max_drawdown(return_stream):
    
    # Cumulative product of portfolio returns
    cumprod_ret = (return_stream + 1).cumprod()*100
        
    # Convert the index in datetime format
    cumprod_ret.index = pd.to_datetime(cumprod_ret.index)
    
    # Define a variable trough_index to store the index of lowest value before new high
    trough_index = (np.maximum.accumulate(cumprod_ret) - cumprod_ret).idxmax()
    
    # Define a variable peak_index to store the index of maximum value before largest drop
    peak_index = cumprod_ret.loc[:trough_index].idxmax()
    
    # Calculate the maximum drawdown using the given formula
    maximum_drawdown = 100 * \
        (cumprod_ret[trough_index] - cumprod_ret[peak_index]) / \
        cumprod_ret[peak_index]    
    
    return maximum_drawdown

def get_annual_returns(return_stream):
    
    # Total number of trading days in a year is 252
    trading_days = 252
    
    # Calculate the average daily returns
    average_daily_returns = return_stream.mean()    
    
    annual_returns = ((1 + average_daily_returns)**(trading_days) - 1) * 100
    
    return annual_returns

def get_skewness(return_stream):
    return(return_stream.skew())

def get_kurtosis(return_stream):
    return(return_stream.kurtosis())
    
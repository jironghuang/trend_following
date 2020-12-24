#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 01:33:08 2020

@author: jirong
"""
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize, rosen, rosen_der

#Function extracted from Robert Carver's pysystemtrade github page
def robust_vol_calc(x,
                    days=35,
                    min_periods=10,
                    vol_abs_min=0.0000000001,
                    vol_floor=True,
                    floor_min_quant=0.05,
                    floor_min_periods=100,
                    floor_days=500,
                    backfill = False):
    """
    Robust exponential volatility calculation, assuming daily series of prices
    We apply an absolute minimum level of vol (absmin);
    and a volfloor based on lowest vol over recent history
    :param x: data
    :type x: Tx1 pd.Series
    :param days: Number of days in lookback (*default* 35)
    :type days: int
    :param min_periods: The minimum number of observations (*default* 10)
    :type min_periods: int
    :param vol_abs_min: The size of absolute minimum (*default* =0.0000000001)
      0.0= not used
    :type absmin: float or None
    :param vol_floor Apply a floor to volatility (*default* True)
    :type vol_floor: bool
    :param floor_min_quant: The quantile to use for volatility floor (eg 0.05
      means we use 5% vol) (*default 0.05)
    :type floor_min_quant: float
    :param floor_days: The lookback for calculating volatility floor, in days
      (*default* 500)
    :type floor_days: int
    :param floor_min_periods: Minimum observations for floor - until reached
      floor is zero (*default* 100)
    :type floor_min_periods: int
    :returns: pd.DataFrame -- volatility measure
    """

    # Standard deviation will be nan for first 10 non nan values
    vol = x.ewm(adjust=True, span=days, min_periods=min_periods).std()
    
    vol[vol < vol_abs_min] = vol_abs_min

    if vol_floor:
        # Find the rolling 5% quantile point to set as a minimum
        vol_min = vol.rolling(
            min_periods=floor_min_periods, window=floor_days).quantile(
                quantile=floor_min_quant)

        # set this to zero for the first value then propagate forward, ensures
        # we always have a value
        vol_min.at[vol_min.index[0]] =  0.0
        vol_min = vol_min.ffill()

        # apply the vol floor
        vol_with_min = pd.concat([vol, vol_min], axis=1)
        vol_floored = vol_with_min.max(axis=1, skipna=False)
    else:
        vol_floored = vol

    if backfill:
        # have to fill forwards first, as it's only the start we want to backfill, eg before any value available
        vol_forward_fill = vol_floored.fillna(method = "ffill")
        vol_backfilled = vol_forward_fill.fillna(method = "bfill")
    else:
        vol_backfilled = vol_floored

    return vol_backfilled


def raw_ewmac(price, Lfast=8, Lslow=None):
    
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback    
    """
    if Lslow is None:
        Lslow=4*Lfast
    
    fast_ewma=price.ewm(span=Lfast).mean()
    slow_ewma=price.ewm(span=Lslow).mean()
    raw_ewmac=fast_ewma - slow_ewma
    
    vol=robust_vol_calc(price.diff()).to_frame()  

    forecast = raw_ewmac.to_frame()/ np.array(vol)
        
    return forecast

def raw_breakout(price, lookback, smooth = None):
    
    """
    Calculate break out trading rule forecast  
    """    
    
    roll_max = price.rolling(lookback).max()
    roll_min = price.rolling(lookback).min()
    roll_mean = (roll_max + roll_min) / 2
    
    forecast = 40 * ( price - roll_mean ) / (roll_max - roll_min)
       
    return forecast

    
def compute_forecast_scalar(xcross, target_abs_forecast=10,window=250000,min_periods=500,backfill=True):
    
    """
    Calculate forecast scalar to bring up raw forecast value to average absolute value of 10
    """     
    
    if xcross.shape[1] == 1:
        x = xcross.abs().iloc[:, 0]
    else:
        x = xcross.ffill().abs().median(axis=1)
    
    avg_abs_value = x.rolling(window=window, min_periods=min_periods).mean()
    scaling_factor = target_abs_forecast / avg_abs_value
    
    return scaling_factor


def compute_norm_forecast(norm_price, forecast_mtd=raw_ewmac, param=8):    
    
    """
    Calculate normalized forecast and cap forecast to between -20 and +20
    """      
          
    raw_forecast = [forecast_mtd(norm_price[i],param) for i in norm_price]
    raw_forecast = pd.concat(raw_forecast, axis=1)    
    forecast_scalar = compute_forecast_scalar(raw_forecast)    
    adj_forecast = raw_forecast.multiply(forecast_scalar, axis="index")    
    
    adj_forecast_cap = adj_forecast.copy()
    adj_forecast_cap[adj_forecast_cap > 20] = 20
    adj_forecast_cap[adj_forecast_cap < -20] = -20
    
    return raw_forecast, forecast_scalar, adj_forecast, adj_forecast_cap

if __name__ == "__main__":   
    pass
    

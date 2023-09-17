#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:57:07 2020

@author: jirong
"""
import pandas as pd
import numpy as np
import time
import random
import os
import datetime as dt
import quandl
import yfinance as yf
import matplotlib.pyplot as plt
import util_futures as uf
import util as ut
from scipy.optimize import minimize, rosen, rosen_der
import json
import multiprocessing as mp
from itertools import product
import pyfolio as pf
import empyrical
import statsmodels.api as sm

class FuturesResearch(object):
    
    # constructor 			  		 			     			  	   		   	  			  	
    def __init__(self, data_path, ewmac_variations, breakout_variations, 
                 optimize_weights_path, forecast_diff_before_rebal, 
                 notion_capital_per_position, fix_capital, commission, 
                 boostrap_sample_size, num_samples_per_period, prop_block_boostrap, 
                 max_annual_volatility, ind_instr_ref_volatility):
        
        """
        Constructor for FuturesResearch class
    
        :param data_path: path to data file (e.g. "./trend_following/quantopian_data/futures_incl_2016.csv")
        :param ewmac_varations: list of ewmac variations (e.g. [8,16,32,64])
        :param breakout_variations: list of breakout variations (e.g. [40,80,160,320])
        :param optimize_weights_path: path to storing weights in a folder ('./optimize_weights') 
        :param forecast_diff_before_rebal: Forecast difference before rebalancing an instrument position in a forecast range of -20 to +20 (e.g. 6.0)
        :param notion_capital_per_position (e.g 20000) (parameter used in study)
        :param fix_capital: (e.g 500000) (parameter not used in study)
        :param commission = 20,
        :param bootsrap_sample_size: Minimum sample size in each boostrap (e.g. 300)
        :param num_samples_per_period: Number of sample extracted from a period (e.g. 25)
        :param prop_block_boostrap: Proportion of data extracted in each bootstrap sample (e.g. 0.25)
        :param max_annual_volatility: Maximum portfolio realized volatility allowed (e.g. 0.15) 
        :param ind_instr_ref_volatility: Referenced volatility level for each instrument (e.g. 0.4)
        :return: returns FutureResearch class
        """         
          
        self.data_path = data_path  
        self.returns = None
        self.price = None
        self.commod = None
        self.ewmac_variations = ewmac_variations
        self.breakout_variations = breakout_variations
        self.forecast_ewmac_info = {}
        self.forecast_breakout_info = {}        
        self.expanding_windows_w_bootstrap_info = None
        self.optimize_weights_path= optimize_weights_path
        self.forecast_diff_before_rebal = forecast_diff_before_rebal
        self.notion_capital_per_position = notion_capital_per_position
        self.fix_capital = fix_capital
        self.commission = commission
        self.boostrap_sample_size = boostrap_sample_size
        self.num_samples_per_period = num_samples_per_period
        self.prop_block_boostrap = prop_block_boostrap
        self.combined_forecasts_all_instruments = None
        self.max_annual_volatility  = max_annual_volatility
        self.ind_instr_ref_volatility  = ind_instr_ref_volatility 
        self.full_forecasts_vol = None        
        self.target_leverage_all_instruments = None
        self.portf_vol_adj_factor = None #Adjustment factor
        self.leverage_adj = None      #Final leverage without forecasts
        self.leverage_adj_final = None #Final leverage use for each instrument with forecasts
        
        #Initialize the extraction of data
        self.get_returns_data()
        self.num_instruments_avail = len(self.price.columns) - self.price.isnull().sum(axis=1)
        self.notion_capital_per_position_w_avail_instr = self.fix_capital/self.num_instruments_avail        
        
        #Initialize computation of forecasts
        self.get_norm_ewmac_info()
        self.get_norm_breakout_info()        
        self.get_combined_forecasts_all_instr()
        
        pass

    def get_returns_data(self):
        
        """
        Obtain returns data from file; convert to price level that starts at 1        
        """
        
        #Read in dataset and normalize it to start from one
        #res = pd.read_csv(self.data_path)
        res = pd.read_csv(self.data_path)        
        res['Date'] = pd.to_datetime(res['Date'], format='%Y-%m-%d')
        res.set_index('Date', inplace=True)    
        self.returns = res.fillna(method = "ffill")  
        self.price = (1 + self.returns).cumprod()            
        self.commod = list(self.price.columns)
              
        pass 
                  
    def get_norm_ewmac_info(self):
        
        """
        Obtain normalized ewmac forecasts scaled to a range of -20 to +20      
        """                
              
        for g in self.ewmac_variations:
            
            self.forecast_ewmac_info[g] = {}
            
            norm_price = [uf.raw_ewmac(self.price[i],g) for i in self.price]
            norm_price = pd.concat(norm_price, axis=1)    
            forecast_scalar = uf.compute_forecast_scalar(norm_price)    
            adj_price_ewmac = norm_price.multiply(forecast_scalar, axis="index")        
            raw_forecast, forecast_scalar, adj_forecast, adj_forecast_cap = uf.compute_norm_forecast(self.price, forecast_mtd=uf.raw_ewmac, param= g)
            
            self.forecast_ewmac_info[g]['norm_price'] = norm_price
            self.forecast_ewmac_info[g]['adj_price_ewmac'] = adj_price_ewmac
            self.forecast_ewmac_info[g]['raw_forecast'] = raw_forecast            
            self.forecast_ewmac_info[g]['forecast_scalar'] = forecast_scalar
            self.forecast_ewmac_info[g]['adj_forecast'] = adj_forecast
            self.forecast_ewmac_info[g]['adj_forecast_cap'] = adj_forecast_cap
            
        pass
    
    def get_norm_breakout_info(self):
        
        """
        Obtain normalized donchian channel forecasts scaled to a range of -20 to +20      
        """          
              
        for g in self.breakout_variations:
            
            self.forecast_breakout_info[g] = {}
            
            norm_price = [uf.raw_breakout(self.price[i],g) for i in self.price]
            norm_price = pd.concat(norm_price, axis=1)    
            forecast_scalar = uf.compute_forecast_scalar(norm_price)    
            adj_price_breakout = norm_price.multiply(forecast_scalar, axis="index")        
            raw_forecast, forecast_scalar, adj_forecast, adj_forecast_cap = uf.compute_norm_forecast(self.price, forecast_mtd=uf.raw_breakout, param= g)
            
            self.forecast_breakout_info[g]['norm_price'] = norm_price
            self.forecast_breakout_info[g]['adj_price_breakout'] = adj_price_breakout
            self.forecast_breakout_info[g]['raw_forecast'] = raw_forecast            
            self.forecast_breakout_info[g]['forecast_scalar'] = forecast_scalar
            self.forecast_breakout_info[g]['adj_forecast'] = adj_forecast
            self.forecast_breakout_info[g]['adj_forecast_cap'] = adj_forecast_cap
            
        pass    
        

    def compute_neg_sharpe(self, allocs_wts_forecasts,adj_forecast_single_instrument,price_series,ind_vol_target=0.4):
           
        """
        Compute sharpe in each bootstrap optimization    
        :param allocs_wts_forecasts: np.array weights applied to returns from individual forecasts.
        :param adj_forecast_single_instrument: Normalized forecast time series for each instrument
        :param price_series: Price series of instrument
        :param ind_vol_target: Reference individual volatility target level (e.g. 0.4)
        """         
        
        price_series_returns = price_series.pct_change(1)              
        day_vol = uf.robust_vol_calc(price_series_returns)
        vol = (day_vol * np.sqrt(252)).shift(1) # annualise        
        leverage = ind_vol_target/vol
                    
        #Compute sharpe of individual rules including commission fees   
        def get_returns_rule(rule):
            adj_forecast_single_instrument_rebal = adj_forecast_single_instrument[rule].copy()
            adj_forecast_single_instrument_rebal[:] = np.nan
            
            val_ref = 0    
            for i in range(adj_forecast_single_instrument.shape[0]):
            
                val = adj_forecast_single_instrument.loc[:,rule][i]      
                        
                if abs(val-val_ref) > self.forecast_diff_before_rebal:
                    val_ref = val
                    adj_forecast_single_instrument_rebal[i] = 1         
            
            forecast = pd.concat([adj_forecast_single_instrument[rule], adj_forecast_single_instrument_rebal], axis=1)
            forecast.columns = ['forecast', 'rebal']
            forecast['forecast_rebal'] = np.where(forecast['rebal'] == 1, forecast['forecast'], np.nan)
            forecast['forecast_rebal'] = forecast['forecast_rebal'].ffill().shift(1)
            forecast = pd.concat([forecast, price_series_returns, vol, leverage], axis=1)        
            forecast.columns = ['forecast', 'rebal', 'forecast_rebal', 'price_change', 'vol', 'leverage']    
            forecast['strat_ret'] = (forecast['forecast_rebal']/10) *  forecast['price_change'] * forecast['leverage']
            
            after_return_perc = 1-(self.commission/self.notion_capital_per_position)
            
            forecast['strat_ret_net_cost'] = np.where(forecast['rebal']==1,forecast['strat_ret'] * after_return_perc,forecast['strat_ret'])
            
            #print(forecast['strat_ret_net_cost'].head())                
                
            return forecast['strat_ret_net_cost']
            #return forecast['strat_ret']
        
        col_names = ['ewmac' + str(i) for i in self.ewmac_variations] + ['breakout' + str(i) for i in self.breakout_variations]        
        
        returns = [get_returns_rule(i) for i in col_names]
        returns = pd.concat(returns, axis=1)        
        returns.columns = col_names
                
        weighted_returns = returns.multiply(allocs_wts_forecasts)
        strategy_returns = weighted_returns.sum(axis=1,skipna=False).to_frame() 
        sharpe = ut.get_sharpe(strategy_returns)        
        
        return -sharpe
    

    def optimize_sharpe_single_instrument_period(self, commod, period, bootstrap_index):
        
        """
        Optimize sharpe in each bootstrap optimization and return dictionary of weights and performance. Optimize weight for each isntrument for each bootstrap sample saved as json file    
        :param commod: Commodity symbol
        :param period: Indexes referenced to a dictionary with reference to period which bootstrap indexes are extracted
        :param boostrap_index: Indexes referenced to a dictionary with reference to bootstrap indexes referenced to self.price dataframe
        """         

        print('commod: ' + commod +  
              ' period: ' + str(period) + 
              ' bootstrap_index: ' + str(bootstrap_index)
              )
                   
        try:            
            
            start_index = self.expanding_windows_w_bootstrap_info[period]['bootstrap_index']['start_index'][bootstrap_index]
            end_index = self.expanding_windows_w_bootstrap_info[period]['bootstrap_index']['end_index'][bootstrap_index]
            
            ewmac_forecast = [self.forecast_ewmac_info[i]['adj_forecast_cap'][commod] for i in self.ewmac_variations]
            breakout_forecast = [self.forecast_breakout_info[i]['adj_forecast_cap'][commod] for i in self.breakout_variations]
            
            forecasts = ewmac_forecast + breakout_forecast
            
            col_names = ['ewmac' + str(i) for i in self.ewmac_variations] + ['breakout' + str(i) for i in self.breakout_variations]
            
            adj_forecast_single_instrument = pd.concat(forecasts, axis=1)            
            adj_forecast_single_instrument.columns = col_names  
            price_series = self.price[commod]    
            
            adj_forecast_single_instrument = adj_forecast_single_instrument.iloc[start_index:end_index,]
            price_series = price_series.iloc[start_index:end_index,]
                           
            price_series = price_series.dropna()
            adj_forecast_single_instrument = adj_forecast_single_instrument.dropna()
            
            lb_ub = [(0.0,1.0) for i in adj_forecast_single_instrument.columns]   
            weights_constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) })    
            initial = 1.0/adj_forecast_single_instrument.shape[1]
            allocs_wts_forecasts = np.asarray([initial] * adj_forecast_single_instrument.shape[1])   
            
            result = minimize(self.compute_neg_sharpe, 
                                allocs_wts_forecasts, 
                                args = (adj_forecast_single_instrument,price_series,), 
                                method = 'SLSQP', 
                                bounds = lb_ub, 
                                constraints = weights_constraints,
                                options={'disp':True})    
            
            #print(commod + ' '+ str(result.x) + ' ' + str(result.fun))
            
            #Write result to dictionary
            res_dict = {'commod': commod,\
                       'period': period,\
                       'bootstrap_index': bootstrap_index,\
                       'sharpe': -result.fun[0],\
                       'weights':  (",".join(str(x) for x in np.round(result.x,4)))
                       }
            
            #Write results to text file
            with open(self.optimize_weights_path + '/' + commod + '_' + str(period) + '_' + str(bootstrap_index) + '.json', 'w') as fp:
                json.dump(res_dict, fp)
                
            return res_dict
                
        except:
            
             res_dict = {'commod': commod,\
            'period': period,\
            'bootstrap_index': bootstrap_index,\
            'sharpe': None,\
            'weights':  None}    
                 
             return res_dict                 
                           
    def avg_optimized_sharpe_allinstr_single_period(self, period):
        
        """
        Parallelize optimization of sharpe across instruments in a period    
        :param period: Indexes referenced to a dictionary with reference to period which bootstrap indexes are extracted
        """         
        
        bootstrap_iter = list(range(len(self.expanding_windows_w_bootstrap_info[1]['bootstrap_index']['start_index'])))
        list_parameters = []
        
        #Create list of tuple        
        for i in self.commod:
            
            if len(list_parameters) != 0:            
                
                print(list_parameters)
                with mp.Pool(mp.cpu_count()-4) as pool:
                    results = pool.starmap(self.optimize_sharpe_single_instrument_period, list_parameters)  
            
            #Initialize list parameters for new commod
            list_parameters = []                
                
            for j in bootstrap_iter:
                list_parameters.append(tuple([i,period,j]))     
              
        pass
    
    def get_opt_weight_file(self,file_name,path='optimize_weights/'):
        
        """
        Obtain optimized weight for single file produced by method avg_optimized_sharpe_allinstr_single_period
        :param period: Indexes referenced to a dictionary with reference to period which bootstrap indexes are extracted
        """         
            
        column_names = ["filename", "commod", "period", "boostrap_index", "sharpe", 'comb_weights']
        df_indiv = pd.DataFrame(columns = column_names)    
        
        with open(path + file_name) as f:
            data = json.load(f)
    
        df_indiv = df_indiv.append({'filename': file_name, 
                                    'commod': data['commod'],
                                    'period': data['period'],
                                    'bootstrap_index': data['bootstrap_index'],
                                    'sharpe': data['sharpe'],
                                    'comb_weights': data['weights']
                                    }, 
                                   ignore_index=True)         
        
        col_names = ['ewmac' + str(i) for i in self.ewmac_variations] + ['breakout' + str(i) for i in self.breakout_variations]            
        df_indiv[col_names] = df_indiv['comb_weights'].str.split(',', expand=True)
        
        return df_indiv    
    
    def get_all_opt_weights(self,path='optimize_weights/'):
        
        """
        Obtain optimized weight for all files produced by method avg_optimized_sharpe_allinstr_single_period
        """                
        
        filenames = os.listdir(path)
        data_files = [self.get_opt_weight_file(i) for i in filenames]
        
        comb_files = pd.concat(data_files)
        
        return comb_files
    
    #Dataframe of weights
    #Default of equal weights
    def get_combined_forecasts_single_instr(self, commod, allocs_wts_forecasts=None):
       
        """
        Obtain combined forecasts for single instrument
        :param commod: Commodity symbol
        :param allocs_wts_forecasts: np.array forecast weights. If none, equal weights are assigned to each forecast rule 
        """        
                
        #Get forecasts
        ewmac_forecast = [self.forecast_ewmac_info[i]['adj_forecast_cap'][commod] for i in self.ewmac_variations]
        breakout_forecast = [self.forecast_breakout_info[i]['adj_forecast_cap'][commod] for i in self.breakout_variations]
        
        forecasts = ewmac_forecast + breakout_forecast            
        col_names = ['ewmac' + str(i) for i in self.ewmac_variations] + ['breakout' + str(i) for i in self.breakout_variations]
        
        adj_forecast_single_instrument = pd.concat(forecasts, axis=1)            
        adj_forecast_single_instrument.columns = col_names         
        
        #Weighted forecasts
        if allocs_wts_forecasts is None:
            
            initial = 1.0/adj_forecast_single_instrument.shape[1]
            allocs_wts_forecasts = np.asarray([initial] * adj_forecast_single_instrument.shape[1])        
        
        weighted_forecasts = adj_forecast_single_instrument.multiply(allocs_wts_forecasts)
        weighted_forecasts = weighted_forecasts.sum(axis=1,skipna=False).to_frame() 
        weighted_forecasts.columns = [commod]
        
        return weighted_forecasts
    
    def get_combined_forecasts_all_instr(self, allocs_wts_forecasts=None): 
        
        """
        Obtain combined forecasts for all instruments
        :param allocs_wts_forecasts: np.array forecast weights. If none, equal weights are assigned to each forecast rule 
        """             
        
        commod_names = self.price.columns
        
        if allocs_wts_forecasts is None:
            combined_forecasts_all_instruments = [self.get_combined_forecasts_single_instr(i) for i in commod_names]                        
        
        if not(allocs_wts_forecasts is None):
            combined_forecasts_all_instruments = [self.get_combined_forecasts_single_instr(i, allocs_wts_forecasts) for i in commod_names] 
        
        combined_forecasts_all_instruments = pd.concat(combined_forecasts_all_instruments, axis=1)
        
        self.combined_forecasts_all_instruments = combined_forecasts_all_instruments
        
        return combined_forecasts_all_instruments
    
    def compute_optimal_leverage_all_instruments(self):
        
        """
        Obtain optimal leverage scaled to portfolio target and individual forecasts
        """           
        
        #Obtain returns of original price series
        price_series_returns = self.price.pct_change(1)              
        price_series_forward_returns = price_series_returns.shift(1)              
            
        #Get volatility of returns
        vol_series_non_adj = [uf.robust_vol_calc(price_series_forward_returns.iloc[:,i]) for i in range(len(list(price_series_forward_returns.columns)))]
        vol_series_non_adj = pd.concat(vol_series_non_adj, axis=1) 
        vol_series_non_adj.columns = self.price.columns       
        leverage_series = self.ind_instr_ref_volatility/(vol_series_non_adj * (252**0.5))
        
        #Compute volatility based on forecasts -->max forecasts
        full_forecasts_all_instruments = self.combined_forecasts_all_instruments.copy()
        full_forecasts_all_instruments[full_forecasts_all_instruments>0] = 20
        full_forecasts_all_instruments[full_forecasts_all_instruments<0] = -20        
        full_forecasts_all_instruments_forward = full_forecasts_all_instruments.ffill().shift(1)    
            
        #Find max volatility based on full forecasts
        full_forecasts_returns = leverage_series * full_forecasts_all_instruments_forward/20 * price_series_forward_returns 
        full_forecasts_avg_returns = full_forecasts_returns.mean(axis=1)
        full_forecasts_vol = uf.robust_vol_calc(full_forecasts_avg_returns)
        full_forecasts_vol = full_forecasts_vol * (252**0.5)
        
        #Adjustment factor because of diversification factor
        self.portf_vol_adj_factor = self.max_annual_volatility/full_forecasts_vol        
        self.full_forecasts_vol = full_forecasts_vol
        self.leverage_adj = leverage_series.multiply(self.portf_vol_adj_factor, axis = 'index')   #against 400000/20 position
        
        #Return key output of target leverage for each instrument
        self.leverage_adj_final = self.leverage_adj * ((self.combined_forecasts_all_instruments.shift(1))/20)
        
        self.target_leverage_all_instruments = self.leverage_adj_final
               
        pass
    
    def get_commod_returns(self, commod):
                        
        """
        Obtain returns for instrument based on optimal leverage scaled to portfolio target and individual forecasts
        """          
        
        #Obtain returns
        price_series_returns = self.price.pct_change(1)     
        
        #Form rebalancing, returns after rebal, returns in dollar terms
        leverage_adj_rebal = self.target_leverage_all_instruments[commod].copy()
        leverage_adj_rebal[:] = np.nan
        commod_returns = price_series_returns[commod].copy()
        commod_returns[:] = np.nan
        dollar_returns = price_series_returns[commod].copy()        
        dollar_returns[:] = np.nan
        combined_forecasts_all_instruments_forward = self.combined_forecasts_all_instruments[commod].shift(1).copy()
        
        #Initialize parameters
        current_pos = 0.00001      #dollar terms
        current_lev = 0.00000001   #leverage terms
        risk_capital = self.notion_capital_per_position 
        last_forecasts = 0
        
        for i in range(leverage_adj_rebal.shape[0]):
        
            optimal_lev = self.target_leverage_all_instruments.loc[:,commod][i]   #Used if difference by this much  
            returns = price_series_returns.loc[:,commod][i]
            optimal_forecasts = combined_forecasts_all_instruments_forward[i]               
            
            if(abs(optimal_forecasts-last_forecasts) >= self.forecast_diff_before_rebal):
                leverage_adj_rebal[i] = 1                #rebal at end of last period day
                commod_returns[i] = returns * optimal_lev
                dollar_returns[i] = risk_capital * commod_returns[i] - self.commission
                commod_returns[i] = dollar_returns[i]/ risk_capital 
                current_lev = optimal_lev                   #Use for next period
                current_pos = optimal_lev * risk_capital
                last_forecasts = optimal_forecasts

            elif(abs(optimal_forecasts-last_forecasts) < self.forecast_diff_before_rebal):                
                leverage_adj_rebal[i] = 0                
                commod_returns[i] = returns * current_lev
                dollar_returns[i] = risk_capital * commod_returns[i] 
                current_lev = (risk_capital * current_lev + dollar_returns[i])/risk_capital
                current_pos = current_lev * risk_capital
                
        return commod_returns, leverage_adj_rebal, dollar_returns
    
    def get_all_commod_returns(self):
        
        """
        Obtain returns for all instruments based on optimal leverage scaled to portfolio target and individual forecasts
        """                  
        
        #Trigger optimal leverage mtd
        self.compute_optimal_leverage_all_instruments()  
        
        #Obtain all commod returns
        commod_returns = [self.get_commod_returns(i)[0] for i in list(self.price.columns)]    
    
        commod_returns = pd.concat(commod_returns, axis=1) 
        commod_returns.columns = self.price.columns
        
        return commod_returns
       
    def TSMOM_single_instr_monthly_returns(self, ret, lookback = 12, cost = 0.012):
        
        """
        Obtain TSMOM returns for each single instrument based on 40% realized volatility
        """                  
        
        #Obtain returns           
        ret = ret.ffill()
        ret.dropna(inplace=True)
        
        std_index = ret.resample('BM').last().index
        mth_index = pd.DataFrame(index=std_index)    
        
        ret_index = (1 + ret).cumprod()   #Converting to normalized price series
        ret_index[0] = 1   
        
       
        # equation (1) ex ante vol estimate
        day_vol = ret.ewm(ignore_na=False,
                          adjust=True,
                          com=60,   
                          min_periods=0).std(bias=False)
        vol = day_vol * np.sqrt(261) # annualise
        
        ret_index = pd.concat([ret_index, vol], axis=1)
        ret_index.columns = ['price', 'vol']        
        ret_m_index = ret_index.resample('BM').last().ffill()
            
        mth_index = pd.concat([mth_index, ret_m_index], axis=1) 
        mth_index['returns'] = mth_index['price'].pct_change(1)        
        mth_index['leverage'] = 0.4/mth_index['vol']     
        mth_index['price_change'] = mth_index['price'].pct_change(lookback)    
        mth_index['signal'] = np.where(mth_index['price_change'] > 0, 1, -1)
        mth_index['signal'] = np.where(np.isnan(mth_index['price_change']), np.nan, mth_index['signal'])        
        mth_index['signal_forward'] = mth_index['signal'].shift(1)  
        mth_index['strat_ret'] = mth_index['signal_forward'] *  mth_index['returns'] * mth_index['leverage'] 
        mth_index['strat_ret_after_costs'] = mth_index['signal_forward'] *  mth_index['returns'] * mth_index['leverage'] - (cost/12) 
        
        return mth_index    
 
    def TSMOM_all_instr_returns(self):
        
        """
        Obtain TSMOM returns for all instrument based on 40% realized volatility for each single instrument
        """                          
        
        #Obtain returns
        res = self.price.pct_change(1)           
        ticker_list = list(res.columns)
        
        ret_stream = [self.TSMOM_single_instr_monthly_returns(res[ticker])['strat_ret'] for ticker in ticker_list]     
        ret_stream = pd.concat(ret_stream, axis=1)
        ret_stream.columns = self.price.columns
        
        ret_stream_costs = [self.TSMOM_single_instr_monthly_returns(res[ticker])['strat_ret_after_costs'] for ticker in ticker_list] 
        ret_stream_costs = pd.concat(ret_stream_costs, axis=1)       
        ret_stream_costs.columns = self.price.columns
        
        avg_ret_stream = ret_stream.mean(axis = 1)
        avg_ret_stream_costs = ret_stream_costs.mean(axis = 1)
        
        return ret_stream, ret_stream_costs, avg_ret_stream, avg_ret_stream_costs 
    
    
    def select_period(self, df, start_date, end_date, index_date = 'date'):
        
        """
        Select period in self.price data frame based on starting, ending date or indexes. indexes used in study
        :param start_date: start date
        :param end_date: end date
        :param index_date: select by 'index' or 'date'        
        """            
        
        if not(type(df) is pd.core.frame.DataFrame or type(df) is pd.Series):
            raise ValueError('Time series input matrix must be of type pd.core.frame.DataFrame or pd.Series')            
        if not(index_date == 'date' or index_date == 'index'):
            raise ValueError('index_date should be string "date" or "index"')
                    
        try:    
            if index_date == 'date':                
                sel_df = df[start_date:end_date]                
            elif index_date == 'index':    
                sel_df = df[start_date:end_date]
                                                               
        except:
            print('Error with selecting df with %s' %(start_date, end_date))           
        
        return sel_df

    def extract_boostrap_periods(self, df, num_samples = 10, start_sample_index = 0, end_sample_index = None, sample_size = 300, prop_block_boostrap = 0.25):

        """
        Function for selecting period
        
        :param df: Data-frame
        :param num_samples: Number of block samples
        :param start_sample_index: Start of sample index        
        :param end_sample_index: End of sample index   
        :param sample_size: Minimum sample size length   
        :param prop_block_boostrap: Proportion of data used in each sample        
        :return: returns dictionary of start and end indexes
        """    
           
        #Select proportion size if larger
        sample_size_prop = round(end_sample_index * prop_block_boostrap)
        
        if sample_size_prop > sample_size:
            sample_size = sample_size_prop
        
        if end_sample_index is None:
            end_sample_index = df.shape[0] - sample_size - 1
        else:
            end_sample_index = end_sample_index - sample_size - 1              
    
        def rand_int(seed_num):   
            random.seed(seed_num)        
            return random.randrange(start_sample_index, end_sample_index)
        
        start_index = np.array([rand_int(n) for n in range(num_samples)])
        end_index = start_index + sample_size        
        
        return {'start_index': start_index, 'end_index': end_index}        
    
    def create_window_index(self, df, window = 'expanding', days_block = 252):
        
        """
        Method for creating window index
        
        :param df: Data-frame
        :param window: expanding or sliding
        :param days_block: testing block size which is also used to create multiple of training block size
        :return: returns list of training and testing indexes
        """      
        
        num_blocks = int(df.shape[0] / 252)
        
        if window == 'expanding':
            windows_index = [[[0,(days_block * (n+1) - 1)], [days_block * (n+1), days_block * (n+1) + days_block - 1]] \
                             if n<(num_blocks-2) else\
                                 [[0,(days_block * (n+1) - 1)], [days_block * (n+1), (df.shape[0]-1)]] \
                                     for n in range(num_blocks-1)]               
                
        windows_index.pop(0)   #Ignore first period                
        
        return windows_index
    
    def create_dictionary_window_n_bootstrap_index(self,read_pickle = False):
        
        """
        Method for creating dictionar of window and bootstrap indexes.
        """            
        
        window_index = self.create_window_index(self.price, window = 'expanding', days_block = 252)
        
        self.expanding_windows_w_bootstrap_info = {}
        length = 0
        for n in window_index:
            length += 1
            self.expanding_windows_w_bootstrap_info[length] = {}
            self.expanding_windows_w_bootstrap_info[length]['in_sample_index'] = n[0]    
            self.expanding_windows_w_bootstrap_info[length]['out_sample_index'] = n[1]       
            bootstrap_index = self.extract_boostrap_periods(self.price, num_samples = self.num_samples_per_period, start_sample_index = 0, end_sample_index = n[0][1], sample_size=self.boostrap_sample_size, prop_block_boostrap = self.prop_block_boostrap)   
            self.expanding_windows_w_bootstrap_info[length]['bootstrap_index'] = bootstrap_index        
        pass    
            
if __name__ == "__main__":  
    pass

#!/usr/bin/env python
# coding: utf-8

# In[18]:


#import libraries
import pandas as pd      
import numpy as np   
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt


#Import data
data_source = yf.Ticker('^GSPC')


#check data
data_hist = data_source.history(period='max')
data_hist=data_hist.dropna()
data = data_hist["1980-01-01":"2009-12-31"].copy()


#calculate returns

def calc_returns(srs, offset=1):
    returns = pd.Series(dtype='float64', index=srs.index)
    for i in range (offset, len(srs)):
            returns.iloc[i] = (srs.iloc[i]-srs.iloc[i-offset])/srs.iloc[i-offset] 
    
    return returns
    
data["daily_returns"] = calc_returns(data["Close"])
data["next_day_returns"] = data["daily_returns"].shift(-1)
data.head()



# data preparation
VOL_THRESHOLD = 5  
data["srs"] = data["Close"]

SMOOTH_WINDOW = 252 


ewm = data["srs"].ewm(halflife=SMOOTH_WINDOW)
means = ewm.mean()
stds = ewm.std()

ub = means + VOL_THRESHOLD * stds
data["srs"] = np.minimum(data["srs"], ub);

lb = means - VOL_THRESHOLD * stds
data["srs"] = np.maximum(data["srs"], lb);

data["daily_returns"] = calc_returns(data["srs"],1)

plt.plot(data["daily_returns"]);

def rescale_to_target_volatility(srs,vol=0.15):
    return srs *  vol / srs.std() / np.sqrt(252)

def plot_captured_returns(next_day_captured, plot_with_equal_vol = None):
    """ Parameters:
            next_day_captured: time-series of next day returns
        Return:
            matplotlib.pyplot of cumulative returns """
    
    if plot_with_equal_vol is not None:
        srs = rescale_to_target_volatility(next_day_captured.copy(),vol=plot_with_equal_vol)
    else:
        srs = next_day_captured.copy()
        
    ((srs.shift(1) + 1).cumprod() - 1).plot()
    plt.ylabel("Cumulative  returns");

    
captured_returns_longonly = data['next_day_returns']["1990-01-01":]
plot_captured_returns(captured_returns_longonly)


#calculate performance metrics Sortino, Sharpe, Calmer ratio, Max drawdown

def calc_downside_deviation(srs):
    """ Parameters:
            srs: pandas time-series
        Return:
            Downside Deviation (defined above) """
    negative_returns = srs.apply(lambda x: x if x < 0 else np.nan).dropna() * np.sqrt(252)
    return negative_returns.std()

def calc_max_drawdown(srs):
    """ Parameters:
            srs: pandas time-series
        Return:
            MDD (defined above) """
    cumulative_max = srs.cummax()
    drawdown = cumulative_max - srs
    return drawdown.max()

def calc_profit_and_loss_ratio(srs):
    """ Parameters:
            srs: pandas time-series
        Return:
            PnL ratio (defined above) """
    return np.mean(srs[srs>0])/np.mean(np.abs(srs[srs<0]))

def calculate_statistics(srs, print_results=True):
    """ Parameters:
            srs: pandas time-series
            print_results: bool to print statistics
        Return:
            Metrics and risk adjusted performance metrics (defined above) """
    
    risk_free_rate = 0.04
   
    mean = srs.mean()
    vol = srs.std()
    
    returns_annualised =  mean*252
    vol_annualised = vol*np.sqrt(252)
    downside_devs_annualised = calc_downside_deviation(srs)
    max_drawdown = calc_max_drawdown(srs)
    pnl_ratio = calc_profit_and_loss_ratio(srs)
    perc_positive_return = len(srs[srs>0])/len(srs)
    
    sharpe = (returns_annualised - risk_free_rate) / vol_annualised
    sortino = (returns_annualised - risk_free_rate) / downside_devs_annualised
    calmar = returns_annualised / max_drawdown
    
    if print_results:
        print("\033[4mPerformance Metrics:\033[0m")
        print(f"Annualised Returns = {returns_annualised:.2%}")
        print(f"Annualised Volatility = {vol_annualised:.2%}")
        print(f"Downside Deviation = {downside_devs_annualised:.2%}")
        print(f"Maximum Drawdown = {max_drawdown:.2%}")
        print(f"Sharpe Ratio = {sharpe:.2f}")
        print(f"Sortino Ratio = {sortino:.2f}")
        print(f"Calmar Ratio = {calmar:.2f}")
        print(f"Percentage of positive returns = {perc_positive_return:.2%}")
        print(f"Profit/Loss ratio = {pnl_ratio:.3f}")
   
    return {
        "returns_annualised":  returns_annualised,
        "vol_annualised": vol_annualised,
        "downside_deviation_annualised": downside_devs_annualised,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "pnl_ratio": pnl_ratio,
    }
stats_longonly = calculate_statistics(captured_returns_longonly)

VOL_LOOKBACK = 40  
VOL_TARGET = 0.10  

def volatility_scaled_returns(daily_returns, vol_lookback = VOL_LOOKBACK, vol_target = VOL_TARGET):
    """ 
    Parameters:
        daily_returns: pandas time-series of the daily returns
        print_results: bool to print statistics
    Return:
        Volatility scaled returns for annualised VOL_TARGET of 15% 
    """
    

    vol_actual = daily_returns.ewm(span=vol_lookback).std() * np.sqrt(252)    
    
    scaled_returns = (vol_target/vol_actual) * daily_returns
    plt.plot(scaled_returns)
    plt.plot(daily_returns)
    return scaled_returns


data['scaled_returns'] = volatility_scaled_returns(data["daily_returns"])
print(f"Signal annualised volatility: {data['scaled_returns'].std()*np.sqrt(252):.2%}")

data["trading_rule_signal"] = (1 + data["scaled_returns"]).cumprod()
data["scaled_next_day_returns"] = data["scaled_returns"].shift(-1)

captured_returns_volscaled_lo = data["scaled_next_day_returns"]["1990-01-01":]

plot_captured_returns(captured_returns_longonly, plot_with_equal_vol = VOL_TARGET)

plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
plt.legend(["Unscaled", "Vol. scaled"]);
stats_volscaled_longonly = calculate_statistics(captured_returns_volscaled_lo)
data["annual_returns"] = calc_returns(data["srs"], 252)
# With the calculated returns, adjust your trading position accordingly
captured_returns_volscaled_tsmom = (
    np.sign(data["annual_returns"])*data["scaled_next_day_returns"]
)["1990-01-01":]

# Plot the time series momentum strategy against the long-only strategy
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_tsmom, plot_with_equal_vol = VOL_TARGET)
plt.legend(["Long Only", "TSMOM"])
stats_volscaled_tsmom = calculate_statistics(captured_returns_volscaled_tsmom)



# Define timescales in trading days
timescales = [5, 21, 63, 126]  # approximating 1 week, 1 month, 1 quarter, half a year

# Create a dictionary to store stats for each timescale
stats_volscaled_tsmom_timescales = {}

for timescale in timescales:
    # Calculate returns over the specific timescale
    data[f"{timescale}_returns"] = calc_returns(data["srs"], timescale)

    # Adjust your trading position accordingly
    captured_returns_volscaled_tsmom = (
        np.sign(data[f"{timescale}_returns"])*data["scaled_next_day_returns"]
    )["1990-01-01":]

    # Plot the time series momentum strategy against the long-only strategy
    plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
    plot_captured_returns(captured_returns_volscaled_tsmom, plot_with_equal_vol = VOL_TARGET)
    plt.legend(["Long Only", f"TSMOM {timescale} days"])

    # Calculate statistics
    stats_volscaled_tsmom_timescales[timescale] = calculate_statistics(captured_returns_volscaled_tsmom)
    


# Define timescales in trading days
timescales = [21, 252]  # 1 month and 1 year

# Define weights
weights = [0.0, 0.25, 0.5, 0.75, 1.0]

# Create a dictionary to store stats for each combination of timescales and weights
stats_combined_signals = {}

data['252_returns'] = calc_returns(data["srs"], 252)

for w in weights:
    # Calculate combined signal
    data["combined_signal"] = w * np.sign(data[f"{timescales[0]}_returns"]) +                               (1 - w) * np.sign(data[f"{timescales[1]}_returns"])

    # Adjust your trading position accordingly
    captured_returns_combined_signal = (
        data["combined_signal"] * data["scaled_next_day_returns"]
    )["1990-01-01":]

    # Plot the combined signal strategy against the long-only strategy
    plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
    plot_captured_returns(captured_returns_combined_signal, plot_with_equal_vol = VOL_TARGET)
    plt.legend(["Long Only", f"Combined signal (w = {w})"])

    # Calculate statistics
    stats_combined_signals[w] = calculate_statistics(captured_returns_combined_signal)

# Define position sizing function
def phi(y):
    return y * np.exp(-(y ** 2) / 4) / 0.89

# Short and long trend combinations used for MACD
MACD_TREND_COMBINATIONS = [(8, 24), (16, 48), (32, 96)]

class MACDStrategy:
    def __init__(self, trend_combinations=None):
        # Setting parameters
        self.x_vol_window = 63
        self.y_vol_window = 252

        if trend_combinations is None:
            self.trend_combinations = MACD_TREND_COMBINATIONS
        else:
            self.trend_combinations = trend_combinations

    def compute_indiv_signal(self, prices, short_window, long_window):
        # Compute individual signals

        # Compute trend over short timescale
        short_trend = prices.ewm(alpha=1.0/short_window).mean()
        # Compute trend over long timescale
        long_trend = prices.ewm(alpha=1.0/long_window).mean()

        x = short_trend - long_trend
        y = x / prices.rolling(self.x_vol_window).std().fillna(method="bfill")
        z = y / y.rolling(self.y_vol_window).std().fillna(method="bfill")
        return z

    def get_signal(self, prices):
        # Calculate combined signal
        trend_combinations = self.trend_combinations
        signal_df = None
        
        for short_window, long_window in trend_combinations:

            indiv_signal = self.compute_indiv_signal(prices, short_window, long_window)

            if signal_df is None:
                signal_df = phi(indiv_signal)
            else:
                signal_df += phi(indiv_signal)

        return signal_df / len(trend_combinations)
# Calculate returns using MACD
captured_returns_volscaled_macd = (
    MACDStrategy().get_signal(data["srs"])*data["scaled_next_day_returns"]
)["1990-01-01":]

# Plot cumulative returns of MACD vs long only
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_macd, plot_with_equal_vol = VOL_TARGET)
plt.legend(["Long Only", "MACD"]);

# Calculate returns using different MACD filters
MACD_returns = {}

# MACD_TREND_COMBINATIONS list is already defined in your code
for trend_comb in MACD_TREND_COMBINATIONS:
    MACD_returns[trend_comb] = (
        MACDStrategy([trend_comb]).get_signal(data["srs"])*data["scaled_next_day_returns"]
    )["1990-01-01":]

# Plot the captured returns for each MACD filter
plt.figure(figsize=(15,10))
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)

for trend_comb, returns in MACD_returns.items():
    plot_captured_returns(returns, plot_with_equal_vol = VOL_TARGET)

plt.legend(["Long Only"] + [str(trend_comb) for trend_comb in MACD_TREND_COMBINATIONS])
plt.title("Comparison of MACD filters")
plt.show()

correlation = captured_returns_volscaled_lo.corr(captured_returns_volscaled_tsmom)
print(f"Correlation between the strategies: {correlation}")
# Build portfolio
portfolio_returns = (captured_returns_volscaled_lo + captured_returns_volscaled_tsmom) / 2

# Plot the returns of the individual strategies and the portfolio
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_tsmom, plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(portfolio_returns, plot_with_equal_vol=VOL_TARGET)
plt.legend(["Long Only", "TSMOM", "Portfolio"])
plt.title('Comparison of Strategies and Portfolio Returns')
plt.show()


# In[ ]:





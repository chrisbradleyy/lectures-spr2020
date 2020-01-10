import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#from scipy import stats

aapl = pdr.get_data_yahoo('AAPL', start=2006)
plt.plot(aapl['Adj Close'])

# plot y=log ret , x=ret
aapl['ret']=aapl['Adj Close'].pct_change() 
aapl['log_ret'] = np.log(aapl['Adj Close'].pct_change()+1)
plt.plot(aapl['ret'].sort_values(),aapl['log_ret'].sort_values()) # I was going to make a particular point, but clearly there is an outlier 
plt.plot( [-.2,.15],[-.2,.15] )
plt.title("log ret =/= ret")

# plot daily returns
plt.plot(aapl['log_ret'])
plt.plot(aapl['log_ret'].rolling(window=30).sum()) # smooth with rolling window

# plot monthly returns 
monthly_log_ret = aapl['log_ret'].resample('M').sum()
plt.plot(monthly_log_ret)

# monthly ret vrs monthly rolling ret
plt.plot(monthly_log_ret)
plt.plot(aapl['log_ret'].rolling(window=30).sum()) # smooth with rolling window

# dist of daily rets
aapl['log_ret'].hist(bins=50)
print(aapl['log_ret'].describe())

# cumulative returns from Jan 2000
cum_daily_ret = (1 + aapl['ret']).cumprod()
cum_daily_ret.plot()

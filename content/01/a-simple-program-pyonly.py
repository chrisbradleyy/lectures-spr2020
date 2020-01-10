import pandas as pd
import numpy as np
import pandas_datareader as pdr # pip install pandas_datareader if needed
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# get stock returns for 3 stocks (NOTE: THIS IGNORES DIVIDENDS)

stock_prices = pdr.get_data_yahoo(['AAPL','MSFT','VZ'], start=2006)
stock_prices = stock_prices.filter(like='Adj Close') # reduce to just columns with this in the name
stock_prices.columns = ['AAPL','MSFT','VZ']

daily_pct_change = pd.DataFrame()
for stock in ['AAPL','MSFT','VZ']:
    daily_pct_change[stock] = stock_prices[stock].pct_change() 

print(daily_pct_change.describe())

###############################################################################
# data viz - compare the return distribution of 3 firms
###############################################################################

# we need this helper function for a plot

def plot_unity(xdata, ydata, **kwargs):
    '''
    Adds a 45 degree line to the pairplot for plots off the diagonal
    
    Usage: 
    grid=sns.pairplot( <call pairplot as you want >  )
    grid.map_offdiag(plot_unity)
    '''
    mn = min(xdata.min(), ydata.min())
    mx = max(xdata.max(), ydata.max())
    points = np.linspace(mn, mx, 100)
    plt.gca().plot(points, points, color='k', marker=None,
            linestyle='--', linewidth=1.0)
     
# compare the return distribution of 3 firms visually...
    
grid = sns.pairplot(daily_pct_change,diag_kind='kde',kind="reg")
grid.map_offdiag(plot_unity) # how cool is that!

###############################################################################
# get the factor loadings
###############################################################################

# get FF factors merged with the stock returns
ff = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily',start=2006)[0] # the [0] is because the imported obect is a dictionary, and key=0 is the dataframe
ff.rename(columns={"Mkt-RF":"mkt_excess"}, inplace=True) # cleaner name
ff = ff.join(daily_pct_change,how='inner') # merge with stock returns
for stock in ['MSFT','AAPL','VZ']:    
    ff[stock] = ff[stock] * 100 # FF store variables as percents, so convert to that
    ff[stock+'_excess'] = ff[stock] - ff['RF'] # convert to excess returns in prep for regressions
print(ff.describe()) # ugly...
pd.set_option('display.float_format', lambda x: '%.2f' % x) # show fewer digits
pd.options.display.max_columns = ff.shape[1] # show more columns
print(ff.describe(include = 'all')) # better!

# run the models- 
params=pd.DataFrame()
for stock in ['MSFT','AAPL','VZ']:        
    print('\n\n\n','='*40,'\n',stock,'\n','='*40,'\n')
    model = sm.formula.ols(formula = stock+"_excess ~ mkt_excess + SMB + HML", data = ff).fit()
    print(model.summary())
    params[stock] = model.params.tolist()
params.set_index(model.params.index,inplace=True)   
    
pd.set_option('display.float_format', lambda x: '%.4f' % x) # show fewer digits
print(params)

###############################################################################
# plot some Apple stock returns
###############################################################################

grid = sns.pairplot(daily_pct_change,diag_kind='kde',kind="reg")
grid.map_offdiag(plot_unity) # how cool is that!
plt.show(grid)

# plot the cummulative returns... NOTE: excludes dividends and splits!
cumrets=(daily_pct_change+1).cumprod()-1
plt.clf() # clear the prior plot before starting a new one
sns.lineplot(data=cumrets).set_title("Returns, ex-dividends")
plt.show(grid)


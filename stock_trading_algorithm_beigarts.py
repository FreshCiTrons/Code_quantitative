# Installing necessary packages, for all packages using python (not Jupyter) use pip install -r requirements.txt

from statsmodels.regression.rolling import RollingOLS
from sklearn.cluster import KMeans
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.ticker as mtick
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')

# Getting the s&p500 list stock symbols from wiki and making it pretty

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()

# Assigning start dates and end dates to retrieve info from YahooFinance
end_date = '2024-05-31'
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*5)

# Retrieving data from yahoo finance

data_file = yf.download(tickers=symbols_list,
                        start=start_date,
                        end=end_date)

# Formatting the retrieved data

data_file = data_file.stack()
data_file.index.names = ['date','ticker']
data_file.columns = data_file.columns.str.lower()

# Saving the retrieved data as csv to escape less errors later
# data_file.to_csv('data/sp500.csv')

# Calculating the features and technical indicators
# Starting with Garman-Klass volatility
# Formula can be found here - https://i.imgur.com/b9hCVEl.png
# Volatility measure of a given asset

data_file['garman klass'] = ((np.log(data_file['high']) - np.log(data_file['low']))**2)/2 - (2*np.log(2)-1) * ((np.log(data_file['adj close']) - np.log(data_file['open']))**2)

# Now moving to Relative Strength Index
# Formula - https://i.imgur.com/WoyNAe9.png ,we will be using built in from pandas_ta library though

data_file['rsi'] = data_file.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
#data_file.xs('GOOGL', level=1)['rsi'].plot()

# Bollinger bands
# Formula for upper and lower band - https://i.imgur.com/rY0FlSo.png

data_file['bb_low'] = data_file.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
data_file['bb_mid'] = data_file.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])                                                          
data_file['bb_high'] = data_file.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

# Reformating extra collumns for ATR - Average True Range, as the formula uses 3 collumns and not one,
# the transform in pandas ta won't work and the result will be need to be formated before adding to the collumn.
# Therefore we are writing our own function for that, to manage the resulted data

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
              low=stock_data['low'],
              close=stock_data['close'],
              length=14)
    return atr.sub(atr.mean()).div(atr.std())
# Adding the ATR collumn
data_file['atr'] = data_file.groupby(level=1, group_keys=False).apply(compute_atr)

# Now when calculating MACD, it is quite the same, we need to write our own function for the same reasons
# We are doing this in general to not worry later, as we are going to be using a machine learning model.
# And we only want to feed it proper data 
# Formula - https://i.imgur.com/7Rx8BsY.png


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
data_file['macd'] = data_file.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

# Now calculating the dollar volume in millions

data_file['dollar_volume'] = (data_file['adj close']*data_file['volume'])/1e6

# Make it to a monthly level and filter out the top 150 most liquid stocks for each month
# We do this to reduce training time
# Here we will adjust all the indicators
# For all the indicators except the dollar volume we take the last data in the month

last_cols = [c for c in data_file.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                                 'high', 'low', 'close']]
data = (pd.concat([data_file.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   data_file.unstack()[last_cols].resample('M').last().stack('ticker')],
                   axis=1)).dropna()

# Now to calculate 5 year rolling average of dollar volume for each of the stocks before filtering out

data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())
data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

# Presenting the <150
# print(data)
data_save = data.to_csv('data/150_most_liquid_stocks.csv')


# Calculating monthly returns for assesing monthly horizons/momentum

# skipping outliers (0.5%)
outlier_cutoffs = 0.005

# To capture time series dynamics that reflect, for example, momentum patterns, we compute historical returns 
# using the method .pct_change(lag), that is, returns over various monthly periods - 1,2,3,6,9,12
# We want to do that to reflect the momentum patterns for each stock

def calculate_returns(data_file):
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        data_file[f'return_{lag}m'] = (data_file['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoffs),
                                                    upper=x.quantile(1-outlier_cutoffs)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
    return data_file

# Formatting and presenting the data with returns 
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()
#print(data)

# Here we introduce the Fama-French factors. This is done to estimate the exposure of our assets to common market risks
# Ex, market, size, value, operating profitability and investment, source - https://arno.uvt.nl/show.cgi?fid=153547#:~:text=The%20important%20Fama%2DFrench%205,one%20of%20them%20is%20momentum + https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

# Retrieving the factors
factors = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

# Formatting the factors
factors.index = factors.index.to_timestamp()
factors = factors.resample('M').last().div(100)
factors.index.name = 'date'
factors = factors.join(data['return_1m']).sort_index()

# We are filtering out the stocks with less than 10 months of data

observations = factors.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factors = factors[factors.index.get_level_values('ticker').isin(valid_stocks.index)]

print(factors)

# Off to calculating rolling factor betas

rolling_betas = ((factors.groupby(level=1,
                                  group_keys=False)
                .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                            exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                            window=min(24,x.shape[0]),
                                            min_nobs=len(x.columns)+1)
                .fit(params_only=True)
                .params
                .drop('const', axis=1))
))

# Now for the values that we just calculated, we used the returns at the end of the month, 
# which means that we wouldn't know this value at the start of the month, we have to 
# shift this data before adding to our other data factors

# rolling_betas.groupby('ticker').shift()

# now let's add it to our data factors
data = (data.join(rolling_betas.groupby('ticker').shift()))
# print(data)

# Now we want to format the data to get rid of NaN's, we do that by replacing them with the Columns average

calculated_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data.loc[:, calculated_factors] = data.groupby('ticker', group_keys=False)[calculated_factors].apply(lambda x: x.fillna(x.mean()))
# print(data)

# Now we remove all the error rows/collumns that might cause problems later

data = data.drop('adj close', axis=1) #Read next comment
data = data.dropna()
#data.info()

# we dont really need the adj close collumn anymore, so we drop it (I put it before dropna to remove anything that might accur)

# Now we are heading down to Machine learning and will try to predict which stocks to keep in the long term based on grouping
# We will be using the K-means clustering algorithm - source - https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
# At first we use random clustering

# def clusters(data):
#     data['cluster'] = KMeans(n_clusters=4,
#                              random_state=0,
#                              init='random').fit(data).labels_
#     return data
# data = data.dropna().groupby('date', group_keys=False).apply(clusters)
# #print(data)

# #Now to visualise the clustering job better, is the exact reason why we didnt normalise rsi
# #Let's use atr & rsi first, but before that, we define a plotting function

# def plot_clusters(data):
#     cluster_0 = data[data['cluster']==0]
#     cluster_1 = data[data['cluster']==1]
#     cluster_2 = data[data['cluster']==2]
#     cluster_3 = data[data['cluster']==3]

#     plt.scatter(cluster_0.iloc[:,5], cluster_0.iloc[:,1], color='red', label='cluster 0')
#     plt.scatter(cluster_1.iloc[:,5], cluster_1.iloc[:,1], color='green', label='cluster 1')
#     plt.scatter(cluster_2.iloc[:,5], cluster_2.iloc[:,1], color='blue', label='cluster 2')
#     plt.scatter(cluster_3.iloc[:,5], cluster_3.iloc[:,1], color='black', label='cluster 3')

#     plt.legend()
#     plt.show()
#     return

# plt.style.use('ggplot')

# for i in data.index.get_level_values('date').unique().tolist():
#     p = data.xs(i, level=0)

#     plt.title(f'Date {i}')

#     plot_clusters(p)

# From this we can assume that clusters move on the rsi scale, which means that the clusters on top (the most upward momentum) change sometimes
# The strategy we are willing to test is what if we invest into the stocks with the highest momentum monthly, will this trading strategy work?
# For this strategy we've decided to stick to the cluster that will be closer to 65-75 rsi, but for this strategy
# random initialisation will not work. What we have to do now is to help the Kmeans clustering algorithm by supplying
# the initial centroids.
# For the initial centres we're taking the RSI indicators.

target_rsi = [30, 45, 55, 70]
initial_centres = np.zeros((len(target_rsi), 18))
# adding target rsi's in rsi collumn of the array created ^
initial_centres[:, 1] = target_rsi

# Let's plot again, but first, we need to remove the pre-assigned clusters
# data = data.drop('clusters', axis=1)

# Plotting the clusters with centroids

def clusters(data):
    data['cluster'] = KMeans(n_clusters=4,
                             random_state=0,
                             init=initial_centres).fit(data).labels_
    return data


data = data.dropna().groupby('date', group_keys=False).apply(clusters)

# Plotting function
def plot_clusters(data):
    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,5], cluster_0.iloc[:,1], color='red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,5], cluster_1.iloc[:,1], color='green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,5], cluster_2.iloc[:,1], color='blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,5], cluster_3.iloc[:,1], color='black', label='cluster 3')

    plt.legend()
    plt.show()
    return

# Plotting loop for every cluster recalc
plt.style.use('ggplot')
for i in data.index.get_level_values('date').unique().tolist():
    p = data.xs(i, level=0)

    plt.title(f'Date {i}')

    plot_clusters(p)

# Now we know that cluster 3 is going to be the one with the stocks which have the most
# upward momentum
# Here we're selecting the assets based on their clustering and forming a portfolio
# based on Efficient Frontier max sharpe ratio optimisation (formula = Sharpe Ratio = (Rx – Rf) / StdDev Rx)
# Efficient Frontier - https://www.investopedia.com/terms/e/efficientfrontier.asp#:~:text=The%20efficient%20frontier%20is%20the,for%20the%20level%20of%20risk.

filtered_df = data[data['cluster']==3].copy()
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index+pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    

# So far we have stocks with best upward momentum on the first date of each month
# Let's define the portfolio optimisation function. 
# We are going to use the portfolio opt package and the efficient frontier (see above: line 280)

# Now knowing that - a)we hope that momentum from last month continues onto this month b) we know which stocks have this momentum now
# c) we need to now understand how much weight do we allocate to each stock
# For optimisation purposes we are having a maximum of 10% weight for a single stock

def optimize_weights(prices, lower_bound=0):
    
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    
    weights = ef.max_sharpe()
    
    return ef.clean_weights()

# Now let's optimise.
# First we need fresh prices

stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])


# Now that we have the new data, we need to calculate the daily returns for each of the stocks that could land in our portfolio

returns_dataframe = np.log(new_df['Adj Close']).diff()
returns_dataframe_save = returns_dataframe.to_csv('data/returns_150.csv')
# print(returns_dataframe)
portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():

    try:
        end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]

        optimisation_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimisation_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')

        # Implementing the weights into the optimisation
        filtered_df = new_df.loc[optimisation_start_date:optimisation_end_date]
        optimisation_df = filtered_df['Adj Close'][cols]
        #print(optimisation_df,'opt test')

        success = False
        try:
            # Calculating the weights
            weights = optimize_weights(prices=optimisation_df,
                            lower_bound=round(1/(len(optimisation_df.columns)*2),3))
            weights = pd.DataFrame(weights, index=pd.Series(0))

            success = True
        except:
            # Handling exceptions
            print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')

        if success == False:
            weights = pd.DataFrame([1/len(optimisation_df.columns) for i in range(len(optimisation_df.columns))],
                                   index=optimisation_df.columns.tolist(),
                                   columns=pd.Series(0)).T
            
        temp_df = returns_dataframe[start_date:end_date]
        temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                   .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                          left_index=True,
                          right_index=True)\
                   .reset_index().set_index(['Date', 'Ticker']).unstack().stack()
        
        temp_df.index.names = ['date', 'ticker']
        temp_df['weighted_return'] = temp_df['return']*temp_df['weight']
        temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')
        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)


    except Exception as e:
        print(e)

weights_save = weights.to_csv('data/weights_portfolio_150.csv')
portfolio_df = portfolio_df.drop_duplicates()
# print(portfolio_df)

# Now that we have the weighted portfolio, we download the s&p500 to compare the returns

sp500 = yf.download(tickers='SPY',
                    start='2021-01-01',
                    end=dt.date.today())
#Calculating the returns
sp500_ret = np.log(sp500[['Adj Close']]).diff().dropna().rename({'Adj Close':'SP500 Buy&Hold'}, axis=1)
# print(sp500_ret)

# Here we plot both the portfolio and s&p500 returns
portfolio_df = portfolio_df.merge(sp500_ret,
                                  left_index=True,
                                  right_index=True)
print(portfolio_df)

#Now that we'we merged the portfolios, we plot them

plt.style.use('ggplot')
portfolio_cumulative_ret = np.exp(np.log1p(portfolio_df).cumsum())-1
portfolio_cumulative_ret[:'2024-04-30'].plot(figsize=(16,6))

plt.title('Kmeans cluster returns over time vs s&p500')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SARIMAX


# Load CSV and set index to be monthly times
bike = pd.read_csv('./data/rides_monthly_aggregate.csv', usecols = [1])
bike.set_index(pd.period_range('1/1/2013', freq='M', periods=84), inplace = True)

# See if data frame was created properly
bike.head()

# Plotting data along with ACF and PACF plots to see any seasonality/trend
bike.plot(figsize = (10,6))

fig, axes = plt.subplots(1, 2, figsize=(15,6))
fig = sm.graphics.tsa.plot_acf(bike, lags=48, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(bike, lags=48, ax=axes[1])


# Fit a Holt-Winters model with additive trend and seasonality to our data
fit_hw = ExponentialSmoothing(bike['num_rides'], seasonal_periods = 12, trend = 'add', seasonal = 'add').fit()

# Plotting the H-W model's fitted values alongside the true data
bike_plot = bike['num_rides'].plot(figsize = (10,6), title = "Holt-Winters' Method Bike Share Fit")
bike_plot.set_ylabel("Number of Bike rentals")
bike_plot.set_xlabel("Year")

fit_hw.fittedvalues.plot(ax = bike_plot, style = '--', color = 'DarkRed')
bike_plot.legend(['Bike Rentals', 'H-W Model Fit'])

# The MAPE (Mean Average Percentage Error) of the H-W model
np.average(np.absolute((fit_hw.fittedvalues - bike['num_rides']) / bike['num_rides']))

# Then for making forecasts with the H-W model
fit_hw.forecast(12)

# Plot the decomposition of the model's states/components
states_decomp = pd.DataFrame(np.c_[fit_hw.level, fit_hw.slope, fit_hw.season], columns=['level','slope','seasonal'], index=bike.index)

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(10,6))
states_decomp[['level']].plot(ax = ax1, title = "Level")
states_decomp[['slope']].plot(ax = ax2, title = "Trend/Slope")
states_decomp[['seasonal']].plot(ax = ax3, title = "Seasonality")






# Trying out the Seasonal ARIMA model
# Since we see clear seasonality in the data, we'll first try season differencing. The seasonally
# differenced data looks much closer to being stationary
bike['seasonal_diff'] = bike.diff(periods = 12)
bike['seasonal_diff'].plot(figsize = (10,6))

# Plotting the original data alongside the seasonally differenced data
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
bike['num_rides'].plot(ax = axes[0], title = "Monthly Bike Rentals")
bike['seasonal_diff'].plot(ax = axes[1], title = "Monthly Bike Rentals - Seasonally Differenced")
axes[1].hlines(0, bike.index[0], bike.index[-1], 'r')

# Plotting PACF and ACF plots of the seasonally differenced data
fig, axes = plt.subplots(1, 2, figsize=(15,6))
fig = sm.graphics.tsa.plot_acf(bike['seasonal_diff'].iloc[12:], lags=36, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(bike['seasonal_diff'].iloc[12:], lags=36, ax=axes[1])





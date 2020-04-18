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
bike['num_rides'].plot(figsize = (10,6))

fig, axes = plt.subplots(1, 2, figsize=(15,6))
fig = sm.graphics.tsa.plot_acf(bike['num_rides'], lags=36, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(bike['num_rides'], lags=36, ax=axes[1])


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

fig, ax = plt.subplots(figsize=(15, 6))
fig = bike['num_rides'].plot(ax = ax, title = "Holt-Winters' Method Bike Share Forecast")
fig.set_ylabel("Number of Bike rentals")
fig.set_xlabel("Year")
fit_hw.forecast(12).plot(ax = ax,style = '--', color = 'DarkRed')
fig.legend(['Bike Rentals', 'H-W Forecast'])





# Trying out the Seasonal ARIMA model
# Since we see clear seasonality in the data, we'll first try season differencing. The seasonally
# differenced data looks much closer to being stationary
bike['seasonal_diff'] = bike.diff(periods = 12)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig = bike['num_rides'].plot(ax = axes[0], title = "Monthly Bike Rentals")
fig = bike['seasonal_diff'].plot(ax = axes[1], title = "Monthly Bike Rentals - Seasonally Differenced")
axes[1].hlines(0, bike.index[0], bike.index[-1], 'r')


# Plotting PACF and ACF plots of the seasonally differenced data
fig, axes = plt.subplots(1, 2, figsize=(15,6))
fig = sm.graphics.tsa.plot_acf(bike['seasonal_diff'].iloc[12:], lags=36, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(bike['seasonal_diff'].iloc[12:], lags=36, ax=axes[1])


fit_sarima = SARIMAX(bike['num_rides'], order = (1,0,1), seasonal_order=(1,1,0,12)).fit()

fit_sarima.resid.plot()


bike_plot = bike['num_rides'].plot(figsize = (10,6), title = "SARIMA Model Bike Share Fit")
bike_plot.set_ylabel("Number of Bike rentals")
bike_plot.set_xlabel("Year")

fit_sarima.fittedvalues.plot(ax = bike_plot, style = '--', color = 'DarkOrange')
bike_plot.legend(['Bike Rentals', 'SARIMA Model Fit'])





# Plotting the SARIMA and H-W forecasts together. We see that the H-W model forecasts greater
# values on average in comparison to the SARIMA
fig, ax = plt.subplots(figsize=(15, 6))
fig = bike['num_rides'].plot(ax = ax, title = "Holt-Winters' Method Bike Share Forecast")
fig.set_ylabel("Number of Bike rentals")
fig.set_xlabel("Year")
fit_hw.forecast(12).plot(ax = ax,style = '--', color = 'DarkRed')
fit_sarima.forecast(12).plot(ax = ax,style = '--', color = 'DarkOrange')
fig.legend(['Bike Rentals', 'H-W Forecast', 'SARIMA Forecast'])
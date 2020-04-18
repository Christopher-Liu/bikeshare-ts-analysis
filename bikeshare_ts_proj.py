import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SARIMAX


# Load CSV and set index to be monthly times
bike = pd.read_csv('./data/rides_monthly_aggregate.csv', usecols = [1])
bike.set_index(pd.period_range('1/1/2013', freq='M', periods=84), inplace = True)

# See if data frame was created properly
bike.head()

# Plotting data to see any seasonality/trend
bike.plot(figsize = (10,6))

# Fit a Holt-Winters model with additive trend and seasonality to our data
fit_hw = ExponentialSmoothing(bike, seasonal_periods = 12, trend = 'add', seasonal = 'add').fit()

fit_hw.params


# Plotting the H-W model's fitted values alongside the true data
bike_plot = bike.plot(figsize = (10,6), title = "Holt-Winters' Method Bike Share Fit")
bike_plot.set_ylabel("Number of Bike rentals")
bike_plot.set_xlabel("Year")

fit_hw.fittedvalues.plot(ax = bike_plot, style = '--', color = 'DarkRed')
bike_plot.legend(['Bike Rentals', 'H-W Model Fit'])



# Plot the decomposition of the model's states/components
states_decomp = pd.DataFrame(np.c_[fit_hw.level, fit_hw.slope, fit_hw.season], columns=['level','slope','seasonal'], index=bike.index)

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(10,6))
states_decomp[['level']].plot(ax = ax1, title = "Level")
states_decomp[['slope']].plot(ax = ax2, title = "Trend/Slope")
states_decomp[['seasonal']].plot(ax = ax3, title = "Seasonality")

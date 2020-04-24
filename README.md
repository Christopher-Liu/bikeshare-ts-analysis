# Bicycle Sharing Time Series Analysis and Forecasting

Bike sharing is a form of transportation in urban areas where passengers can get rent bikes from a number of public stations litered throughout a city for minutes or hours at a time and return them at any public station. Whether for leisurely riding or as a main method for commuting throughout the city, bike sharing has the potential to transform how we move around cities as many densely populated metro areas can frequently be navigated quicker by bike than by car. I performed an end-to-end time series analysis of bicycle rental data from the bike sharing company, Capital Bikeshare. I thought it would be interesting to dive into the data and extract insight into the behavior and trends of riders throughout the years and see how bike sharing has been developing as a major form of public transportation. With the models, I was also interested in performing forecasts of business growth in the upcoming year.


## Methods Used
Time Series analysis:
* Exponential Smoothing (Holt-Winters' Method)
* Seasonal ARIMA


## Technologies
* Python
* statsmodels
* pandas
* NumPy
* matplotlib
* Jupyter
* PostgreSQL


## Project Description

Over 6 years of data was provided by Capital Bikeshare, a bike sharing company that operates in the Washington DC metro area. The data tracks every one of their bikes that have been rented since 2011, the duration of the bike ride, and the starting and ending location of the trips. I downloaded all of the available data since 2013 and aggregated the values into monthly bike rentals in order to analyze the behavior of ridership throughout the years. 

With 6 years worth of data, the full data set grew to be bigger than something that could be handled in-memory with my own computer, so instead of loading everything up into pandas, I utilized PostgreSQL for data loading and aggregation. As a solution, I loaded and aggregated the data in chunks. 

With the data aggregated into monthly values, it could be loaded into pandas for further exploration and analysis. I mainly used matplotlib and statsmodels for all data exploration and analysis purposes. 


## Table of Contents
1. [Data Aggregation and EDA](https://github.com/Christopher-Liu/bikeshare-ts-analysis/blob/master/1-Data_Aggregation_and_EDA.ipynb) 
2. [Modeling: Holt-Winters' Method](https://github.com/Christopher-Liu/bikeshare-ts-analysis/blob/master/2-Modeling_Holt-Winters_Method.ipynb)
3. [Modeling: Seasonal ARIMA](https://github.com/Christopher-Liu/bikeshare-ts-analysis/blob/master/3-Modeling_Seasonal_ARIMA.ipynb)


## Contact
If you have any questions, feel free to reach me via LinkedIn:
https://www.linkedin.com/in/christopher-liu-aa264978/

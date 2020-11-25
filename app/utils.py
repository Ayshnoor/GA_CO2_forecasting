import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import VAR
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
import gzip, pickle, pickletools

from var import VAR_model
from sarima import auto_ARIMA_model
from prophetm import prophet_model

from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.graph_objs as go
import boto3


def VAR_forecast_plot(country, years):
    #setup
    country_for_forecast = VAR_model(str(country), years)
    country_for_forecast.setup()

    #open pickled
    pickled = pickle.load(open(f'VAR_models/{country}_VAR.p', 'rb'))
    pickled_fit = pickled.fit(maxlags=5, ic='aic')
    forecast = pickled_fit.forecast(y=country_for_forecast.new_df.values, steps=years)

    #forecast df
    country_for_forecast.forecast_df = pd.DataFrame(forecast, columns=country_for_forecast.new_df.columns, index = pd.to_datetime([year for year in range(2019, 2019+years)], errors='coerce', format='%Y'))

    #invert the df
    country_for_forecast.invert_transformation()

    #plot results with plotly

    country_for_forecast.forecast_df = country_for_forecast.forecast_df.filter(regex='_forecast$')
    country_for_forecast.forecast_df.columns = [f'{country_for_forecast.country_name}'+'_CO2', f'{country_for_forecast.country_name}'+'_pop', f'{country_for_forecast.country_name}'+'_gdp']

    x_values = country_for_forecast.df.index
    y_values = country_for_forecast.df[str(country)+'_CO2']

    x_val = country_for_forecast.forecast_df.index
    y_val =country_for_forecast.forecast_df[f'{country}'+'_CO2']

    x_values1 = country_for_forecast.df.index
    y_values1 = country_for_forecast.df[str(country)+'_pop']

    x_val1 = country_for_forecast.forecast_df.index
    y_val1 =country_for_forecast.forecast_df[f'{country}'+'_pop']

    x_values2 = country_for_forecast.df.index
    y_values2 = country_for_forecast.df[str(country)+'_gdp']

    x_val2 = country_for_forecast.forecast_df.index
    y_val2 =country_for_forecast.forecast_df[f'{country}'+'_gdp']

    fig = make_subplots(rows=1, cols=3)

    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode = 'lines', name= 'Historical CO2 Emissions (1000 metric tonnes)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_val, y=y_val, mode = 'lines', name= 'Predicted CO2 Emissions (1000 metric tonnes)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_values1, y=y_values1, mode = 'lines', name= 'Historical Population'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_val1, y=y_val1, mode = 'lines', name= 'Predicted Population Growth'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_values2, y=y_values2, mode = 'lines', name= 'Historical GDP Percentage Growth'), row=1, col=3)
    fig.add_trace(go.Scatter(x=x_val2, y=y_val2, mode = 'lines', name= 'Predicted GDP Percentage Growth'),row=1, col=3)

    fig.update_layout(height=600, width=1200, title = f'{country} CO2 Emissions/Population/GDP Forecast Chart')

    return fig

def SARIMA_forecast_plot(country, years):
    #setup
    country_for_forecast = auto_ARIMA_model(str(country), years)
    country_for_forecast.setup()
    
    #get pickle from Amazon S3
    bucket = 'co2models'
    prefix = f'SARIMA_models/{country}_SARIMA.pkl'
    s3 = boto3.resource('s3')
    gzipfile = gzip.open(s3.Bucket(bucket).Object(prefix).get()['Body'], 'rb')
    p = pickle.Unpickler(gzipfile)
    pickled = p.load()

    # pickled = pickle.load(open(f'SARIMA_models/{country}_SARIMA.p', 'rb'))
    forecast = pickled.predict(n_periods=years)
    #forecast df
    country_for_forecast.forecast_df = pd.DataFrame(forecast, columns=[f'{country}' + '_CO2_forecast'],
                                                    index = ([year for year in range(2019, 2019+years)]))
    #Plot results
    print('here')
    x_values = country_for_forecast.df.index
    y_values = country_for_forecast.df[str(country)+'_CO2']
    x_val = country_for_forecast.forecast_df.index
    y_val =country_for_forecast.forecast_df[''.join(list(country_for_forecast.forecast_df.columns))]

    trace1 = go.Scatter(x=x_values, y=y_values, mode = 'lines', name= 'Historical CO2 Emissions (1000 metric tonnes)')
    trace2 = go.Scatter(x=x_val, y=y_val, mode = 'lines', name= 'CO2 Prediction (1000 metric tonnes)')

    data = [trace1, trace2]

    layout = go.Layout(title = f'{country} CO2 Emissions Forecast Chart')

    print('here')

    fig = go.Figure(data=data, layout=layout)

    return fig

def FBProphet_forecast_plot(country, years):
    #setup
    country_for_forecast = prophet_model(str(country), years)
    country_for_forecast.setup()
    
    #get pickle from Amazon S3
    bucket = 'co2models'
    prefix = f'Prophet_models/{country}_Prophet.pkl'
    s3 = boto3.resource('s3')
    gzipfile = gzip.open(s3.Bucket(bucket).Object(prefix).get()['Body'], 'rb')
    p = pickle.Unpickler(gzipfile)
    pickled = p.load()


    #forecast df
    forecast = pd.DataFrame([year for year in range(2019, 2019 + years)], columns=['ds'])
    country_for_forecast.forecast_df = pickled.predict(forecast)

    #plot results (not using the defaults)
    x_values = country_for_forecast.modelling_df['ds']
    y_values = country_for_forecast.modelling_df['y']
    x_val = country_for_forecast.forecast_df['ds']
    y_val =country_for_forecast.forecast_df['yhat']

    trace1 = go.Scatter(x=x_values, y=y_values, mode = 'lines', name= 'Historical CO2 Emissions (1000 metric tonnes)')
    trace2 = go.Scatter(x=x_val, y=y_val, mode = 'lines', name= 'CO2 Prediction (1000 metric tonnes)')

    print('here')

    data = [trace1, trace2]

    layout = go.Layout(title = f'{country} CO2 Emissions Forecast Chart')

    fig = go.Figure(data=data, layout=layout)

    return fig

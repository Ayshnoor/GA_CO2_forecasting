import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates
import statsmodels.api as sm
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
import pickle

class auto_ARIMA_model:
    def __init__(self, country_name, years):
        self.country_name = country_name
        self.years = years
        self.df = pd.read_csv('./data/cleanedDF_modelling.csv')
        self.modelling_df = pd.DataFrame()
        self.new_df = pd.DataFrame() #differenced df
        self.forecast_df = pd.DataFrame()
        self.diff = 1
        self.aic = 0
        self.mse = 0
        self.r2_score = 0

    def setup(self):
        self.df.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
        self.df['Year'] = pd.to_datetime(self.df['Year'], errors='coerce', format='%Y')
        self.df.set_index('Year', inplace=True)
        self.df.sort_index(inplace=True)
        self.df = self.df.loc[(self.df.index < '2019') & (self.df.index > '1949')]
        self.df = self.df.filter(regex='^' + str(self.country_name)+ '_', axis=1)
        self.modelling_df = self.df.filter(regex='CO2$', axis=1)
        self.new_df = pd.DataFrame(index=self.df.index)
        self.stationary_columns()
        return self.modelling_df


    def stationary_columns(self):
        for col in self.modelling_df.columns:
            if adfuller(self.modelling_df[col])[1] < 0.01:
                self.new_df[str(col) + '_diff0'] = self.modelling_df[col]
                self.diff = 0
            elif adfuller(self.modelling_df[col].diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) + '_diff1'] = self.modelling_df[col].diff(1)
                self.diff = 1
            elif adfuller(self.modelling_df[col].diff(1).diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) +'_diff2'] = self.modelling_df[col].diff(1).diff(1)
                self.diff = 2
            elif adfuller(self.modelling_df[col].diff(1).diff(1).diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) +'_diff3'] = self.modelling_df[col].diff(1).diff(1).diff(1)
                self.diff= 3
            elif adfuller(self.modelling_df[col].diff(1).diff(1).diff(1).diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) +'_diff4'] = self.modelling_df[col].diff(1).diff(1).diff(1).diff(1)
                self.diff = 4
            elif adfuller(self.modelling_df[col].diff(1).diff(1).diff(1).diff(1).diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) +'_diff5'] = self.modelling_df[col].diff(1).diff(1).diff(1).diff(1).diff(1)
                self.diff = 5
        self.new_df = self.new_df.dropna()


    def train_model_fit(self): #refernce:https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c
        steps = int(len(self.modelling_df)*0.75)
        train = self.modelling_df.iloc[0:steps]
        test = self.modelling_df.iloc[steps : len(self.modelling_df)]

        stepwise_model = auto_arima(self.modelling_df, start_p=1, start_q=1,max_p=3, max_q=3, m=12, start_P=0, seasonal=True,
                                    d=self.diff, D=1, trace=True, error_action='ignore',suppress_warnings=True, stepwise=True)

        self.aic = stepwise_model.aic()
        stepwise_model.fit(train)
        future_forecast = stepwise_model.predict(n_periods=len(test))
        self.forecast_df = pd.DataFrame(future_forecast, index = test.index, columns=[f'{self.country_name}' +'_CO2_prediction'])
        self.mse = mean_squared_error(test, future_forecast)
        self.r2_score = r2_score(test, future_forecast)
        print(f'Mean Squared Error {self.mse}')
        print(f'R2 Score {self.r2_score}')

        return self.forecast_df

    def model_fit(self):

        stepwise_model = auto_arima(self.modelling_df, start_p=1, start_q=1,max_p=3, max_q=3, m=12, start_P=0, seasonal=True,
                                    d=self.diff, D=1, trace=True, error_action='ignore',suppress_warnings=True, stepwise=True)

        self.aic_total = stepwise_model.aic()
        stepwise_model.fit(self.modelling_df)

        filepath = f'{self.country_name}_SARIMA.pkl'

        with gzip.open(filepath, "wb") as file:
            pickled = pickle.dumps(stepwise_model)
            optimized_pickle = pickletools.optimize(pickled)
            file.write(optimized_pickle)

        #pickle.dump(stepwise_model, open(f'{self.country_name}_SARIMA.p', 'wb'))

        future_forecast = stepwise_model.predict(n_periods=self.years)
        self.forecast_df = pd.DataFrame(future_forecast, index = pd.to_datetime([year for year in range(2019, 2019+self.years)],  errors='coerce', format='%Y'), columns=[f'{self.country_name}' +'_CO2_forecast'])
        return self.forecast_df


    def plot_results (self):

        fig, ax = plt.subplots (figsize = (15, 15))
        sns.lineplot(x = self.df.index.values, y= self.df[str(self.country_name)+'_CO2'])
        sns.lineplot(x = self.forecast_df.index.values, y= self.forecast_df[''.join(list(self.forecast_df.columns))])

        ax.set(xlabel='Date', ylabel ='CO2 Emissions', title = f'CO2 Emissions {self.country_name} (over time)')
        ax.get_xaxis().set_major_locator(mdates.AutoDateLocator())

        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.show();


    def __del__(self):
        print("removing model")

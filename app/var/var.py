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
import pickle
from sklearn.metrics import mean_squared_error, r2_score

class VAR_model:
    def __init__(self, country_name, years):
        self.country_name = country_name
        self.years = years
        self.df = pd.read_csv('./data/cleanedDF_modelling.csv')
        self.new_df = pd.DataFrame() #differenced df
        self.forecast_df = pd.DataFrame(index=pd.to_datetime([year for year in range(2019, 2019 + self.years)],
                                                            errors='coerce', format='%Y'))
        self.mse_CO2 = 0
        self.r2_score_CO2 = 0
        self.mse_pop = 0
        self.r2_score_pop = 0
        self.mse_gdp = 0
        self.r2_score_gdp = 0

    def setup(self):
        self.df.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
        self.df['Year'] = pd.to_datetime(self.df['Year'], errors='coerce', format='%Y')
        self.df.set_index('Year', inplace=True)
        self.df.sort_index(inplace=True)
        self.df = self.df.loc[(self.df.index < '2019') & (self.df.index > '1949')]
        self.df = self.df.filter(regex='^'+ str(self.country_name), axis=1)
        self.new_df = pd.DataFrame(index=self.df.index)
        self.stationary_columns()
        return self.df

    def stationary_columns(self):
        for col in self.df.columns:
            if adfuller(self.df[col])[1] < 0.01:
                self.new_df[col] = self.df[col]
            elif adfuller(self.df[col].diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) + '_diff1'] = self.df[col].diff(1)
            elif adfuller(self.df[col].diff(1).diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) +'_diff2'] = self.df[col].diff(1).diff(1)
            elif adfuller(self.df[col].diff(1).diff(1).diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) +'_diff3'] = self.df[col].diff(1).diff(1).diff(1)
            elif adfuller(self.df[col].diff(1).diff(1).diff(1).diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) +'_diff4'] = self.df[col].diff(1).diff(1).diff(1).diff(1)
            elif adfuller(self.df[col].diff(1).diff(1).diff(1).diff(1).diff(1).dropna())[1] < 0.01:
                self.new_df[str(col) +'_diff5'] = self.df[col].diff(1).diff(1).diff(1).diff(1).diff(1)
            else:
                print(str(col) + '_need_diff')
                print(self.new_df.shape)
        self.new_df = self.new_df.dropna()


    def train_model_fit (self):
        steps = int(len(self.new_df)*0.75)
        train = self.new_df.iloc[0:steps]
        test = self.new_df.iloc[steps: len(self.new_df)]
        model = VAR(train)
        model_fit = model.fit(maxlags = 5, ic = 'aic')
        model_fit.plot_forecast(len(test));
        forecast = model_fit.forecast(y=train.values, steps = len(test))
        self.forecast_df = pd.DataFrame(forecast, index = test.index, columns = train.columns)

        for col in self.forecast_df:
            if 'CO2' in str(col):
                self.mse_CO2 = mean_squared_error(test[str(col)], self.forecast_df[col])
                self.r2_score_CO2 = r2_score(test[str(col)], self.forecast_df[col])
                print(f"mse_CO2: {self.mse_CO2}, r2_score_CO2: {self.r2_score_CO2}")
            if 'pop' in str(col):
                self.mse_pop = mean_squared_error(test[str(col)], self.forecast_df[col])
                self.r2_score_pop = r2_score(test[str(col)], self.forecast_df[col])
                print(f"mse_pop: {self.mse_pop}, r2_score_pop: {self.r2_score_pop}")
            if 'gdp' in str(col):
                self.mse_gdp = mean_squared_error(test[str(col)], self.forecast_df[col])
                self.r2_score_gdp = r2_score(test[str(col)], self.forecast_df[col])
                print(f"mse_gdp: {self.mse_gdp}, r2_score_gdp: {self.r2_score_gdp}")

        for col in self.new_df.columns:
            if '_diff5' in str(col):
                self.forecast_df[str(col) + '_4d_test'] = (self.df[str(col).rstrip('_diff5')].iloc[steps-4] - self.df[str(col).rstrip('_diff5')].iloc[steps-5])+ self.forecast_df[str(col)].cumsum()
                self.forecast_df[str(col) + '_forecast'] = self.df[str(col).rstrip('_diff5')].iloc[steps-1] + self.forecast_df[str(col)+'_4d_test'].cumsum()
            if '_diff4' in str(col):
                self.forecast_df[str(col) + '_3d_test'] = (self.df[str(col).rstrip('_diff4')].iloc[steps-3] - self.df[str(col).rstrip('_diff4')].iloc[steps-4])+ self.forecast_df[str(col)].cumsum()
                self.forecast_df[str(col) + '_forecast'] = self.df[str(col).rstrip('_diff4')].iloc[steps-1]  + self.forecast_df[str(col)+'_3d_test'].cumsum()
            if '_diff3' in str(col):
                self.forecast_df[str(col) + '_2d_test'] = (self.df[str(col).rstrip('_diff3')].iloc[steps-2] - self.df[str(col).rstrip('_diff3')].iloc[steps-3]) + self.forecast_df[str(col)].cumsum()
                self.forecast_df[str(col) + '_forecast'] = self.df[str(col).rstrip('_diff3')].iloc[steps-1]  + self.forecast_df[str(col)+'_2d_test'].cumsum()
            if '_diff2' in str(col):
                self.forecast_df[str(col) + '_1d_test'] = (self.df[str(col).rstrip('diff2').rstrip('_')].iloc[steps-1] - self.df[str(col).rstrip('diff2').rstrip('_')].iloc[steps-2]) + self.forecast_df[str(col)].cumsum()
                self.forecast_df[str(col) + '_forecast'] = self.df[str(col).rstrip('diff2').rstrip('_')].iloc[steps-1]  + self.forecast_df[str(col)+'_1d_test'].cumsum()
            if '_diff1' in str(col):
                self.forecast_df[str(col) + '_forecast'] = self.df[str(col).rstrip('_diff1')].iloc[steps-1] + self.forecast_df[str(col)].cumsum()



    def model_fit(self):
        model = VAR(self.new_df)
        model_fit = model.fit(maxlags= 5, ic='aic')
        forecast = model_fit.forecast(y=self.new_df.values, steps=self.years)
        self.forecast_df = pd.DataFrame(forecast, columns=self.new_df.columns, index = pd.to_datetime([year for year in range(2019, 2019+self.years)],
                                                                                                      errors='coerce', format='%Y'))

        pickle.dump(model, open(f'{self.country_name}_VAR.p', 'wb'))

        self.invert_transformation()



    def invert_transformation (self):
        for col in self.new_df.columns:
            if "_diff5" in str(col):
                self.forecast_df[str(col)+'_4d'] = (self.df[str(col).rstrip('_diff5')].iloc[-4] - self.df[str(col).rstrip('_diff5')].iloc[-5]) + (self.forecast_df[str(col)].cumsum())
                self.forecast_df[str(col)+'_forecast'] = self.df[str(col).rstrip('_diff5')].iloc[-1] + self.forecast_df[str(col)+'_4d'].cumsum()
            if "_diff4" in str(col):
                self.forecast_df[str(col)+'_3d'] = (self.df[str(col).rstrip('_diff4')].iloc[-3] - self.df[str(col).rstrip('_diff4')].iloc[-4]) + self.forecast_df[str(col)].cumsum()
                self.forecast_df[str(col)+'_forecast'] = self.df[str(col).rstrip('_diff4')].iloc[-1] + self.forecast_df[str(col)+'_3d'].cumsum()
            elif "_diff3" in str(col):
                self.forecast_df[str(col)+'_2d'] = (self.df[str(col).rstrip('_diff3')].iloc[-2] - self.df[str(col).rstrip('_diff3')].iloc[-3]) + self.forecast_df[str(col)].cumsum()
                self.forecast_df[str(col)+'_forecast'] = (self.df[str(col).rstrip('_diff3')].iloc[-1]) + self.forecast_df[str(col)+'_2d'].cumsum()
            elif "_diff2" in str(col):
                self.forecast_df[str(col)+'_1d'] = (self.df[str(col).rstrip('diff2').rstrip('_')].iloc[-1] - self.df[str(col).rstrip('diff2').rstrip('_')].iloc[-2]) + self.forecast_df[str(col)].cumsum()
                self.forecast_df[str(col)+'_forecast'] = (self.df[str(col).rstrip('diff2').rstrip('_')].iloc[-1]) + self.forecast_df[str(col)+'_1d'].cumsum()
            elif "_diff1" in str(col):
                self.forecast_df[str(col)+'_forecast'] = self.df[str(col).rstrip('_diff1')].iloc[-1] + self.forecast_df[str(col)].cumsum()
            else:
                self.forecast_df[str(col)+'_forecast'] = self.forecast_df[str(col)]
        return self.forecast_df

    def plot_results (self):
        self.forecast_df = self.forecast_df.filter(regex='_forecast$')
        self.forecast_df.columns = [f'{self.country_name}'+'_CO2', f'{self.country_name}'+'_pop', f'{self.country_name}'+'_gdp']
        fig, ax = plt.subplots (ncols =3, figsize = (15, 15))
        sns.lineplot(x = self.df.index.values, y= self.df[str(self.country_name)+'_CO2'], ax=ax[0])
        sns.lineplot(x = self.forecast_df.index.values, y = self.forecast_df[f'{self.country_name}'+'_CO2'], ax=ax[0])
        sns.lineplot(x = self.df.index.values, y= self.df[str(self.country_name)+'_pop'], ax=ax[1])
        sns.lineplot(x = self.forecast_df.index.values, y = self.forecast_df[f'{self.country_name}'+'_pop'], ax=ax[1])
        sns.lineplot(x = self.df.index.values, y= self.df[str(self.country_name)+'_gdp'], ax=ax[2])
        sns.lineplot(x = self.forecast_df.index.values, y = self.forecast_df[f'{self.country_name}'+'_gdp'], ax=ax[2])

        ax[0].set(xlabel='Date', ylabel ='CO2 Emissions', title = f'CO2 Emissions {self.country_name} (over time)')
        ax[1].set(xlabel='Date', ylabel ='Pop', title = f'Population {self.country_name} (over time)')
        ax[2].set(xlabel='Date', ylabel ='GDP Percentage', title = f'GDP Percentage {self.country_name} (over time)')
        ax[0].get_xaxis().set_major_locator(mdates.AutoDateLocator())
        ax[1].get_xaxis().set_major_locator(mdates.AutoDateLocator())
        ax[2].get_xaxis().set_major_locator(mdates.AutoDateLocator())

        #plt.setp(ax.get_xticklabels(), rotation=45)
        plt.show();

    def __del__(self):
        print("removing model")

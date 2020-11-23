import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import gzip, pickle, pickletools

class prophet_model:
    def __init__(self, country_name, years):
        self.country_name = country_name
        self.years = years
        self.df = pd.read_csv('./data/cleanedDF_modelling.csv')
        self.df_CO2only = pd.DataFrame()
        self.modelling_df = pd.DataFrame() #modelling df
        self.forecast_df = pd.DataFrame([year for year in range(2019, 2019+self.years)], columns=['ds'])
        self.mse = 0
        self.r2_score = 0

    def setup(self):
        self.df.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
        self.df['Year'] = pd.to_datetime(self.df['Year'], errors='coerce', format='%Y')
        self.df = self.df.loc[(self.df['Year'] < '2019') & (self.df['Year']> '1949')]
        self.df_CO2only = self.df.filter(regex='_CO2$', axis=1)
        self.df_CO2only.columns = self.df_CO2only.columns.str.rstrip('_CO2')
        self.modelling_df = pd.concat([self.df['Year'], self.df_CO2only[f'{self.country_name}']], axis=1, keys=['ds','y'])
        return self.modelling_df

    def train_model_forecast_plot(self):
        steps = int(len(self.modelling_df)*0.75)
        train = self.modelling_df.iloc[0:steps]
        test = self.modelling_df.iloc[steps:len(self.modelling_df)]
        model = Prophet()
        model.fit(train)
        self.forecast_df = pd.DataFrame(test['ds'], columns=['ds'])
        self.forecast_df = model.predict(self.forecast_df)
        model.plot(self.forecast_df)

        self.mse = mean_squared_error(test['y'],self.forecast_df['yhat'])
        self.r2_score = r2_score(test['y'], self.forecast_df['yhat'])


    def model_forecast_plot(self):
        model = Prophet()
        model.fit(self.modelling_df)
        self.forecast_df = pd.DataFrame([year for year in range(2019, 2019+self.years)], columns=['ds'])
        self.forecast_df = model.predict(self.forecast_df)

        filepath = f'{self.country_name}_Prophet.pkl'

        with gzip.open(filepath, "wb") as file:
            pickled = pickle.dumps(model)
            optimized_pickle = pickletools.optimize(pickled)
            file.write(optimized_pickle)

        # pickle.dump(model, open(f'{self.country_name}_Prophet.p', 'wb')) <too large files>
        
        model.plot(self.forecast_df)


    def __del__(self):
        print("removing model")

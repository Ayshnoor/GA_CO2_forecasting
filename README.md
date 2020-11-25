# Capstone Project - CO2 Forecasting using Time-Series Analysis
This application intends to describe CO2 emissions on a per-country basis and how this has changed over time. As well as convey a potential forecast of CO2 based on three different types of Time-Series Analysis.
The user can input country and number of years for the forecast.

## Table of Contents
* [Background](Background)
* [Data Source](Data_Source)
* [Modelling Details](Modelling_Details)
* [Technologies](Technologies)
* [Setup](Setup)
* [Next Steps](Next_Steps)

## Background
Considering the significance of climate change and the impact CO2 emissions have on our planet, how do we, as a world, find solutions to reduce the impact of global CO2 emissions?
The thought was to understand which countries are larger contributors to CO2 emissions and create forecasts to describe possible future outcomes. The hope for the web app was to provide a deeper understanding of our current landscape to inspire regional solutions and possible future reductions of CO2 emissions. 

## Data Source
Data sourced from Gapminder.org on a per Country basis. All three models use free CO2 Emissions data from the Carbon Dioxide Information Analysis Center via Gapminder.org (cc-by license).  One particular model, Vector Auto-Regression or VAR, uses open GDP data (as a percentage of growth) from the University of Groningen - Faculty of Economics and Business and uses available Population data from Maddison and the UN both via Gapminder.org (cc-by license).
The data does go back to 1751, but data for modeling purposes was from 1949 to 2018.

## Modelling Details
Three different time-series forecasting models generated nearly 600 predictive models. All forecasts within the app are go-forward projections, meaning they use the whole data set to project future values.

**Vector Autoregression or VAR**: This model is multivariate and also uses GDP (as a percentage of growth) and population. Population growth was chosen as a crucial variable to understand CO2 emissions, thinking that Population growth would directly relate to CO2 emissions. GDP (as a percentage of growth), indicating development, was chosen given a similar logic, where countries with significant GDP growth would also contribute proportionally to CO2 emissions.  You can find more information about Vector Auto-Regression [here](https://en.wikipedia.org/wiki/Vector_autoregression).

**FB Prophet**: This model is univariate and uses only CO2 data for forecasting. This model is an open-source python library that forecasts time series data based on an additive model. More information on the model can be found [here](https://facebook.github.io/prophet/#:~:text=Forecasting%20at%20scale.&text=Prophet%20is%20a%20procedure%20for,daily%20seasonality%2C%20plus%20holiday%20effects.&text=Prophet%20is%20open%20source%20software%20released%20by%20Facebook's%20Core%20Data%20Science%20team.).

**Seasonal Autoregressive Integrated Moving Average or SARIMA**: This model is also univariate and uses only the CO2 data to forecast into the future. More information on this model can be found [here](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average).

## Technologies
* App created using Flask library
* Graphs and World Map generated using Plotly library
* Individual libraries associated with each different model type - VAR uses statsmodel.tsa.api package (VAR), FBProphet uses Prophet from the FBProphet package, and SARIMA uses auto_arima from pmdarima.arima package. Models were pickled and compressed using gzip.


## Web App

Heroku has strict requirements on size given its free nature of hosting web apps. Therefore, Pickles were created for each model and compressed using gzip. AWS S3 was needed to store the large files related to SARIMA and FBProphet. Code included in the utils.py to retreive the compressed pickles and to decompress and use for forecast. 

Please find the web app here: https://co2-forecasting.herokuapp.com/

## Next Steps
This web app is step one, but future steps include the following:
* Aggregation of forecast to create a regional or global projection
* Recommendation on how to reduce CO2 emissions in various regions (based on possible projections)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.offline as offline
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def CO2_graph():

    df_CO2 = pd.read_csv('data/CO2_Emissions_to_Plot.csv')
    df_CO2.drop('Unnamed: 0', axis=1, inplace=True)

    data_slider = []

    x = [col for col in df_CO2.columns if col != 'index']

    for year in x:
        data_each_year = dict (type='choropleth',
                              locations = df_CO2['index'],
                              locationmode = 'country names',
                              z= df_CO2[year].astype(float),
                              colorscale = 'Viridis',
                              colorbar= {'title': 'CO2 emissions per year (1000 metric tonnes)'})
        data_slider.append(data_each_year)

    steps=[]

    for i in range(len(data_slider)):
        step = dict(method='restyle',
                   args=['visible', [False] * len(data_slider)],
                   label='Year {}'.format(i + 1751))
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

    layout = dict(title ='Historical CO2 Emissions from the Burning of Fossil Fuels since 1751', geo=dict(showframe=False,
                                                              showcoastlines=False,
                                                              projection_type='equirectangular'),
                                                              sliders=sliders)

    fig = dict(data=data_slider, layout=layout)

    return fig

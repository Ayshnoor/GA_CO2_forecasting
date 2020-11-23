from flask import Flask, render_template, session, redirect, url_for, request
from plotly.offline import plot
from plotly.graph_objs import *
from flask import Markup
from co2_chart import CO2_graph
from utils import FBProphet_forecast_plot, VAR_forecast_plot, SARIMA_forecast_plot
from plotly.subplots import make_subplots

app = Flask(__name__)


@app.route('/')
def index():

    my_plot_div = plot(CO2_graph(), output_type='div')

    return render_template('Index.html', div_placeholder=Markup(my_plot_div))

@app.route('/VAR')
def var():
    return render_template('VAR_dash.html') #context object would have country

@app.route('/SARIMA')
def sarima():
    return render_template('SARIMA_dash.html')

@app.route('/FBProphet')
def FBProphet():
    return render_template('FBProphet_dash.html')


@app.route('/FBProphet/<country>')
def FBProphet_country(country):
    if request.args.get('number_years'):
            years = int(request.args.get('number_years'))
            my_plot_div = plot(FBProphet_forecast_plot(country, years), output_type='div')
            return render_template('FBProphet_dash.html', country=country, years = years,
                                        div_placeholder = Markup(my_plot_div))
    else:
        return render_template('FBProphet_dash.html',country=country)

@app.route('/VAR/<country>')
def VAR_country(country):
    if request.args.get('number_years'):
        years=int(request.args.get('number_years'))
        my_plot_div = plot(VAR_forecast_plot(country,years), output_type='div')
        return render_template('VAR_dash.html', country=country, years=years,
                              div_placeholder=Markup(my_plot_div))
    else:
        return render_template('VAR_dash.html', country=country)

@app.route('/SARIMA/<country>')
def SARIMA_country(country):
    if request.args.get('number_years'):
        years = int(request.args.get('number_years'))
        my_plot_div = plot(SARIMA_forecast_plot(country, years), output_type='div')
        return render_template('SARIMA_dash.html', country=country, years=years,
        div_placeholder=Markup(my_plot_div))

    else:
        return render_template('SARIMA_dash.html', country=country)


if __name__ == '__main__':
    app.run(debug = True)

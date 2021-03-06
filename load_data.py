import io
import requests
from datetime import datetime
from dateutil.parser import parse as parsedate
from pytz import utc
import pandas as pd

# area type = 'country', 'region', etc.
# area name = either the name of a country or region, eg. England, South East, etc

def load_england_region_data(url):

    datastr = requests.get(url,
                           allow_redirects=True).text
    data_file = io.StringIO(datastr)
    covid_cases = pd.read_csv(data_file,
                              parse_dates=['Specimen date'],
                              index_col=['Area name', 'Specimen date']).sort_index()

    covid_cases.rename(columns={'Specimen date': 'date',
                            'Cumulative lab-confirmed cases': 'cumulative cases',
                            'Area type': 'area type',
                            'Area name': 'area name'},
                   inplace=True)
    # print(covid_cases.columns)

    # need to rename indexes as well as columns
    covid_cases.index.names = ['area name', 'date']

    covid_cases.drop(['Area code',
                      'Daily lab-confirmed cases',
                      'Change in daily cases',
                      'Previously reported cumulative cases',
                      'Previously reported daily cases',
                      'Change in cumulative cases'],
                     axis=1,
                     inplace=True)

    covid_cases = covid_cases[covid_cases['area type'] == 'Region']
    covid_cases.drop('area type',axis=1,inplace=True)

    # print(regions.columns)

    return covid_cases

def load_uk_countries_data(url, country):

    datastr = requests.get(url,
                           allow_redirects=True).text
    data_file = io.StringIO(datastr)
    covid_cases = pd.read_csv(data_file,
                              parse_dates=['Date'],
                              index_col=['Date']).sort_index()

    # print(covid_cases.columns)

    # need to rename indexes as well as columns
    covid_cases.index.names = ['date']

    covid_cases.drop(['Tests',
                      'Deaths'],
                     axis=1,
                     inplace=True)

    covid_cases.rename(columns={'Date': 'date',
                                'ConfirmedCases': 'cumulative cases'},
                       inplace=True)

    covid_cases['area name'] = country
    covid_cases.set_index(['area name'], inplace=True, append=True)
    covid_cases = covid_cases.reorder_levels(['area name', 'date'], axis=0)

    if country == 'Scotland':
        # covid_cases = covid_cases.loc[(covid_cases.index.get_level_values('date') > '2020-06-14')]
        # Manual correction to historical data before Scotland introduced a correction on 15 June 2020.
        # This may not be correct but it is an estimate of the missed cases before correction
        covid_cases.loc[(covid_cases.index.get_level_values('date') > '2020-05-01') & (
                covid_cases.index.get_level_values('date') < '2020-06-15')] = covid_cases.loc[(covid_cases.index.get_level_values('date') > '2020-05-01') & (
                covid_cases.index.get_level_values('date') < '2020-06-15')] + 2000
    return covid_cases


def load_lombardy_region_data(url):

    datastr = requests.get(url,
                           allow_redirects=True).text
    data_file = io.StringIO(datastr)
    covid_cases = pd.read_json(data_file,
                                convert_dates=['data'])
                                # index_col=['data']).sort_index()

    # print(covid_cases.columns)

    covid_cases.drop(['stato', 'codice_regione', 'lat',
       'long', 'ricoverati_con_sintomi', 'terapia_intensiva',
       'totale_ospedalizzati', 'isolamento_domiciliare',
       'variazione_totale_positivi', 'nuovi_positivi', 'dimessi_guariti',
       'deceduti', 'totale_positivi', 'tamponi', 'casi_testati', 'note_it',
       'note_en'],
                     axis=1,
                     inplace=True)

    covid_cases.rename(columns={'data': 'date',
                                'denominazione_regione': 'region',
                                'totale_casi': 'cumulative cases'},
                       inplace=True)

    covid_cases = covid_cases.loc[covid_cases['region'] == 'Lombardia']
    covid_cases.drop(['region'],
                     axis=1,
                     inplace=True)

    covid_cases['area name'] = 'Lombardy'
    covid_cases.set_index(['date'], inplace=True)
    covid_cases.set_index(['area name'], inplace=True, append=True)
    covid_cases = covid_cases.reorder_levels(['area name', 'date'], axis=0)

    return covid_cases


def load_newyork_region_data(url):

    datastr = requests.get(url,
                           allow_redirects=True).text
    data_file = io.StringIO(datastr)
    covid_cases = pd.read_csv(data_file,
                                parse_dates=['date'],
                                index_col=['state', 'date']).sort_index()

    # print(covid_cases.columns)

    states = covid_cases.groupby('state')
    covid_cases = states.get_group('NY')

    covid_cases.rename(columns={'data': 'date',
                                'denominazione_regione': 'region',
                                'totale_casi': 'cumulative cases'},
                       inplace=True)

    covid_cases = covid_cases.loc[covid_cases['region'] == 'Lombardia']
    covid_cases.drop(['region'],
                     axis=1,
                     inplace=True)

    covid_cases['area name'] = 'Lombardy'
    covid_cases.set_index(['date'], inplace=True)
    covid_cases.set_index(['area name'], inplace=True, append=True)
    covid_cases = covid_cases.reorder_levels(['area name', 'date'], axis=0)

    return covid_cases


def download_only_if_newer(url):
    r = requests.get(url)
    url_time = r.headers['Last-Modified']
    url_date = parsedate(url_time)

    now = utc.localize(datetime.now())

    # print(str(url_date.date()))
    # print(str(now.date()))

    if url_date.date() == now.date():
        print('Updating with new cases from today: ' + url_date.strftime("%d/%m/%y"))
        return True, url_date.date
    else:
        print('Latest data has not been updated yet. Last update was ' + url_date.strftime("%d/%m/%y"))
        print('We won\'t bother processing any further')
        return False, url_date.date

# download_only_if_newer(url='https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv'
# )
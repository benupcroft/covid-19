import io
import requests
from datetime import datetime
from dateutil.parser import parse as parsedate
from pytz import utc
import pandas as pd

# area type = 'country', 'region', etc.
# area name = either the name of a country or region, eg. England, South East, etc

def load_england_data(url):

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

    regions = covid_cases[covid_cases['area type'] == 'Region']

    return regions


def download_only_if_newer(url):
    r = requests.get(url)
    url_time = r.headers['Last-Modified']
    url_date = parsedate(url_time)
    # print("URL Date =", url_date)

    now = utc.localize(datetime.now())
    # print("Now Date =", now)
    # current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)

    if url_date.date == now.date:
        print('Updating with new cases from today')
        return True, url_date.date
    else:
        print('Latest data has not been updated yet. Last update was ' + url_date.strftime("%d/%m/%y"))
        print('We won\'t bother processing any further')
        return False, url_date.date

# download_only_if_newer(url='https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv'
# )
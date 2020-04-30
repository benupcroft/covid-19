import io
import requests
import pandas as pd

# area type = 'country', 'region', etc.
# area name = either the name of a country or region, eg. England, South East, etc

def load_data():
    url = 'https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv'
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

    print('columns = ', regions.columns)
    print('index = ', regions.index)

    return regions

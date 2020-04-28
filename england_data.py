import io
import requests
import pandas as pd


def load_data():
    url = 'https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv'
    datastr = requests.get(url,
                           allow_redirects=True).text
    data_file = io.StringIO(datastr)
    covid_cases = pd.read_csv(data_file,
                              parse_dates=['Specimen date'],
                              index_col=['Area name', 'Specimen date']).sort_index()

    regions = covid_cases[covid_cases['Area type'] == 'Region']
    return regions

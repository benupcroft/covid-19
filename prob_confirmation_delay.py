import os
import requests
import pandas as pd
from matplotlib import dates as mdates
import numpy as np
from matplotlib import pyplot as plt


# Calculate the probability distribution of delay between symptom onset
# and confirmation
def prob_delay():

    # data_file = download_historical_data()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_directory, 'data/linelist.csv')


    # Parse & clean global historical patient info
    patients = parse_patient_info(data_file)

    # Show relationship between onset of symptoms and actual medical confirmation
    plot_onset_vs_confirmation(patients)

    # Calculate the delta in days between onset and confirmation
    delay = (patients.Confirmed - patients.Onset).dt.days

    # Convert samples to an empirical distribution
    p_delay = delay.value_counts().sort_index()
    new_range = np.arange(0, p_delay.index.max()+1)
    p_delay = p_delay.reindex(new_range, fill_value=0)
    p_delay /= p_delay.sum()

    # # Plot the distribution
    fig, axes = plt.subplots(ncols=2, figsize=(9,3))
    p_delay.plot(title='P(Delay)', ax=axes[0])
    p_delay.cumsum().plot(title='P(Delay <= x)', ax=axes[1])
    for ax in axes:
        ax.set_xlabel('days')

    return p_delay


def download_file(url, local_filename):
    """From https://stackoverflow.com/questions/16694907/"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename

def download_historical_data():
    URL = "https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.csv"

    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_directory, 'data/linelist.csv')
    LINELIST_PATH = data_file

    print('Downloading file, this will take a while ~100mb')
    try:
        download_file(URL, LINELIST_PATH)
        # clear_output(wait=True)
        print('Done downloading.')
        return LINELIST_PATH
    except:
        print('Something went wrong. Try again.')


    # if not os.path.exists(LINELIST_PATH):
    #     print('Downloading file, this will take a while ~100mb')
    #     try:
    #         download_file(URL, LINELIST_PATH)
    #         clear_output(wait=True)
    #         print('Done downloading.')
    #     except:
    #         print('Something went wrong. Try again.')
    # else:
    #     print('Already downloaded CSV')


def parse_patient_info(data_file):
    # Load the patient CSV
    patients = pd.read_csv(
        data_file,
        parse_dates=False,
        usecols=[
            'date_confirmation',
            'date_onset_symptoms'],
        low_memory=False)

    patients.columns = ['Onset', 'Confirmed']

    # There's an errant reversed date
    patients = patients.replace('01.31.2020', '31.01.2020')

    # Only keep if both values are present
    patients = patients.dropna()

    # Must have strings that look like individual dates
    # "2020.03.09" is 10 chars long
    is_ten_char = lambda x: x.str.len().eq(10)
    patients = patients[is_ten_char(patients.Confirmed) &
                        is_ten_char(patients.Onset)]

    # Convert both to datetimes
    patients.Confirmed = pd.to_datetime(
        patients.Confirmed, format='%d.%m.%Y')
    patients.Onset = pd.to_datetime(
        patients.Onset, format='%d.%m.%Y')

    # Only keep records where confirmed > onset
    patients = patients[patients.Confirmed >= patients.Onset]
    return patients

def plot_onset_vs_confirmation(patients):
    ax = patients.plot.scatter(
        title='Onset vs. Confirmed Dates - COVID19',
        x='Onset',
        y='Confirmed',
        alpha=.1,
        lw=0,
        s=10,
        figsize=(6,6))

    formatter = mdates.DateFormatter('%m/%d')
    locator = mdates.WeekdayLocator(interval=2)

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(formatter)
        axis.set_major_locator(locator)



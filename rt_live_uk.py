#!/usr/bin/python3.7
"""
# Estimating COVID-19's $R_t$ Daily - UK

Adapted from Kevin Systrom's notebook - April 22 2020

Ben and Isaac Upcroft - 26/4/2020

Model originally built by [Thomas Vladeck](https://github.com/tvladeck) in Stan,
parts inspired by the work over at https://epiforecasts.io/,
lots of help from [Thomas Wiecki](https://twitter.com/twiecki).

Thank you to everyone who helped.
"""

# For some reason Theano is unhappy when I run the GP,
# need to disable future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from load_data import load_england_region_data, download_only_if_newer, load_uk_countries_data
from integrity_check import is_data_current, do_daily_cases_increase
from prob_confirmation_delay import prob_delay
from adjust_for_onset_dates import confirmed_to_onset, adjust_onset_for_right_censorship, plot_adjusted_data
from run_model import run_model
from plot_rtuk import plot_rt


# Find the relationship between onset of symptoms and the delay until
# actual medical confirmation
# ~100mb download (be ... patient!)
p_delay = prob_delay()

### Load country data and convert to regions
england_url = 'https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv'
scotland_url = 'https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-totals-scotland.csv'
wales_url = 'https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-totals-wales.csv'
nireland_url = 'https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-totals-northern-ireland.csv'
england_country_url = 'https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-totals-england.csv'
uk_url = 'https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-totals-uk.csv'

is_updated, url_date = download_only_if_newer(england_url)
# don't bother processing if data is still old
if is_updated:
    # download and load up uk data
    # uk data includes all the tests outside of NHS (accounts for almost a 1/3 of tests)
    # regions0 = load_uk_countries_data(uk_url, 'UK')
    # approximately 4% of people aren't recorded in regional-only data
    regions1 = load_england_region_data(england_url)
    regions2 = load_uk_countries_data(scotland_url, 'Scotland')
    regions3 = load_uk_countries_data(wales_url, 'Wales')
    regions4 = load_uk_countries_data(nireland_url, 'N. Ireland')
    # regions5 = load_uk_countries_data(england_country_url, 'England')
    # regions6 = regions1.groupby(level=[1]).sum()
    # regions6['area name'] = 'England from Regions'
    # regions6.set_index(['area name'], inplace=True, append=True)
    # regions6 = regions6.reorder_levels(['area name', 'date'], axis=0)

    frames = [regions1, regions2, regions3, regions4]
    # frames = [regions1]
    regions = pd.concat(frames)
    # print(regions)


    # Check the integrity of the data
    is_data_current(regions)
    do_daily_cases_increase(regions)


    # # Quick test to see delay curve for a single region
    # # Let's look at the adjusted data for a single region - South East
    # # (contains Oxford)
    # region = 'South East'
    # confirmed = regions.xs(region)['Cumulative lab-confirmed cases'].diff().dropna()
    # # confirmed.tail()
    #
    #
    # # Compute the adjustment using the probability of delay - p_delay
    # onset = confirmed_to_onset(confirmed, p_delay)
    # adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    #
    #
    # # Plot the data for our our single region
    # # plot_adjusted_data(region,confirmed,onset,adjusted)

    run_model(p_delay, regions)

    # Plot charts
    plot_rt()

    # Copy update date to web page


    ### Compare to other analyses
    # compare.compare_epiforecasts()


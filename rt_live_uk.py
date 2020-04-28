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
import numpy as np

import england_data
import integrity_check
import prob_confirmation_delay
import adjust_for_onset_dates
import mcmc_model

from matplotlib import pyplot as plt
from matplotlib import dates as mdates

from IPython.display import clear_output

import shutil
import time

## Load country data and convert to regions

regions = england_data.load_data()

#### Check the integrity of the data
integrity_check.is_data_current(regions)
integrity_check.do_daily_cases_increase(regions)

## Find the relationship between onset of symptoms and the delay until
# actual medical confirmation
## ~100mb download (be ... patient!)
p_delay = prob_confirmation_delay.prob_delay()

## Quick test to see delay curve for a single region
# Let's look at the adjusted data for a single region - South East
# (contains Oxford)
region = 'South East'
confirmed = regions.xs(region)['Cumulative lab-confirmed cases'].diff().dropna()
# confirmed.tail()

# Compute the adjustment using the probability of delay - p_delay
onset = adjust_for_onset_dates.confirmed_to_onset(confirmed, p_delay)
adjusted, cumulative_p_delay = adjust_for_onset_dates.adjust_onset_for_right_censorship(onset, p_delay)

# Plot the data for our our single region
# adjust_for_onset_dates.plot_adjusted_data(region,confirmed,onset,adjusted)


# Let's run all the regions and compute Rt
def create_and_run_model(name, region):
    confirmed = region['Cumulative lab-confirmed cases'].diff().dropna()
    onset = adjust_for_onset_dates.confirmed_to_onset(confirmed, p_delay)
    adjusted, cumulative_p_delay = adjust_for_onset_dates.adjust_onset_for_right_censorship(onset, p_delay)
    return mcmc_model.MCMCModel(name, onset, cumulative_p_delay).run()

models = {}

for region, grp in regions.groupby('Area name'):

    print(region)

    if region in models:
        print(f'Skipping {region}, already in cache')
        continue

    models[region] = create_and_run_model(region,grp.droplevel(0))

### Handle Divergences

# Check to see if there were divergences
n_diverging = lambda x: x.trace['diverging'].nonzero()[0].size
divergences = pd.Series([n_diverging(m) for m in models.values()], index=models.keys())
has_divergences = divergences.gt(0)

print('Diverging states:')
# print(divergences[has_divergences])

# Rerun states with divergences
for region, n_divergences in divergences[has_divergences].items():
    models[region].run()

## Compile Results

results = None

for region, model in models.items():

    df = mcmc_model.df_from_model(model)

    if results is None:
        results = df
    else:
        results = pd.concat([results, df], axis=0)

### Write out to CSV
# Uncomment if you'd like
results.to_csv('data/rt_2020_04_28.csv')

### Render Charts

def plot_rt(name, result, ax, c=(.3,.3,.3,1), ci=(0,0,0,.05)):
    ax.set_ylim(0.5, 1.6)
    ax.set_title(name)
    ax.plot(result['median'],
            marker='o',
            markersize=4,
            markerfacecolor='w',
            lw=1,
            c=c,
            markevery=2)
    ax.fill_between(
        result.index,
        result['lower_90'].values,
        result['upper_90'].values,
        color=ci,
        lw=0)
    ax.axhline(1.0, linestyle=':', lw=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

ncols = 3
nrows = int(np.ceil(results.index.levels[0].shape[0] / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(14, nrows*3),
    sharey='row')

for ax, (region, result) in zip(axes.flat, results.groupby('region')):
    plot_rt(region, result.droplevel(0), ax)

fig.tight_layout()
fig.set_facecolor('w')

plt.show()

# move the old plot and save
timestr = time.strftime("%Y%m%d-%H%M%S")
# print timestr
shutil.move("../../mysite/static/img/latest.png", "../../mysite/static/img/old_plots/rt_plot-"+timestr+".png")

fig.savefig('../../mysite/static/img/latest.png', dpi=fig.dpi)





############
# url = 'https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv'
# abbrev = pd.read_csv(url, index_col=['State'], squeeze=True)

# jhu_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
# jhu = pd.read_csv(jhu_url)

# jhu = jhu.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'])
# jhu = jhu.set_index('Province_State')
# jhu.index = jhu.index.rename('region')
# jhu = jhu.drop([
#     'American Samoa',
#     'Guam',
#     'Northern Mariana Islands',
#     'Puerto Rico',
#     'Virgin Islands',
#     'Diamond Princess',
#     'Grand Princess'])

# jhu.index = pd.Index(jhu.index.to_series().replace(abbrev).values, name='region')
# jhu.columns = pd.to_datetime(jhu.columns)
# jhu = jhu.groupby('region').sum()
# jhu = jhu.stack().sort_index()

# state = 'VT'
# ax = jhu.xs(state).diff().plot(label='JHU', color='k', legend=True, title=state)
# # jhu.xs('AK').diff().rolling(7).mean().plot(ax=ax)
# states.xs(state).positive.diff().plot(ax=ax, figsize=(6,4), linestyle=':', label='Covidtracking', legend=True)
# ax.set_xlim(pd.Timestamp('2020-03-01'),None)

# url = 'https://raw.githubusercontent.com/epiforecasts/covid-regional/3ad63ea1acceb797f0628a8037fc206342d267e7/united-states/regional-summary/rt.csv'

# epf = pd.read_csv(url, parse_dates=['date'])
# epf.region = epf.region.replace(abbrev)
# epf = epf.set_index(['region', 'date']).sort_index()
# epf = epf.drop(['Guam', 'Puerto Rico'])

# epf_lookup = {}
# for idx, grp in epf.groupby('region'):

#     epf_lookup[idx]=grp
#     epf_grp = epf_lookup[state]

#     plot_rt(state, result.droplevel(0), ax)
#     plot_rt(state, epf_grp.droplevel(0), ax, c=(1,0,0,1))
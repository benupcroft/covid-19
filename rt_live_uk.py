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

import os
import pandas as pd

from england_data import load_data
from integrity_check import is_data_current, do_daily_cases_increase
from prob_confirmation_delay import prob_delay
from adjust_for_onset_dates import confirmed_to_onset, adjust_onset_for_right_censorship, plot_adjusted_data
import mcmc_model
from plot_rtuk import plot_rt
import compare


# Find the relationship between onset of symptoms and the delay until
# actual medical confirmation
# ~100mb download (be ... patient!)
p_delay = prob_delay()


# Load country data and convert to regions
regions = load_data()


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


# Let's run all the regions and compute Rt
def create_and_run_model(name, region):
    confirmed = region['cumulative cases'].diff().dropna()
    onset = confirmed_to_onset(confirmed, p_delay)
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    return mcmc_model.MCMCModel(name, onset, cumulative_p_delay).run()

models = {}

for region, grp in regions.groupby('area name'):

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
script_directory = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(script_directory, 'data/latest_results.csv')
results.to_csv(results_file)


### Plot charts
plot_rt()


### Compare to other analyses
# compare.compare_epiforecasts()

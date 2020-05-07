# Let's run all the regions and compute Rt
import os
import pandas as pd
import mcmc_model

from adjust_for_onset_dates import confirmed_to_onset, adjust_onset_for_right_censorship, plot_adjusted_data

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_and_run_model(name, region, p_delay):
    confirmed = region['cumulative cases'].diff().dropna()
    onset = confirmed_to_onset(confirmed, p_delay)
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    return mcmc_model.MCMCModel(name, onset, cumulative_p_delay).run()


def run_model(p_delay, regions):
    models = {}

    for region, grp in regions.groupby('area name'):

        print(region)

        if region in models:
            print(f'Skipping {region}, already in cache')
            continue

        models[region] = create_and_run_model(region, grp.droplevel(0), p_delay)

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

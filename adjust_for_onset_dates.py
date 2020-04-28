import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def confirmed_to_onset(confirmed, p_delay):
    ### Translate Confirmation Dates to Onset Dates
    # Our goal is to translate positive test counts to the dates where they likely
    # occured. Since we have the distribution, we can distribute case counts back
    # in time according to that distribution.
    # To accomplish this, we reverse the case time series, and convolve it using
    # the distribution of delay from onset to confirmation. Then we reverse the
    # series again to obtain the onset curve. Note that this means the data will
    # be 'right censored' which means there are onset cases that have yet to be
    # reported so it looks as if the count has gone down.

    assert not confirmed.isna().any()

    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1],
                       periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)

    return onset


def adjust_onset_for_right_censorship(onset, p_delay):
    ### Adjust for Right-Censoring
    #
    # Since we distributed observed cases into the past to recreate the onset curve,
    # we now have a right-censored time series. We can correct for that by asking
    # what % of people have a delay less than or equal to the time between the day
    # in question and the current day.
    # For example, 5 days ago, there might have been 100 cases onset.
    # Over the course of the next 5 days some portion of those cases will be reported.
    # This portion is equal to the cumulative distribution function of our delay
    # distribution. If we know that portion is say, 60%, then our current count of
    # onset on that day represents 60% of the total. This implies that the total is
    # 166% higher. We apply this correction to get an idea of what actual onset cases
    # are likely, thus removing the right censoring.

    cumulative_p_delay = p_delay.cumsum()

    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)

    # Add ones and flip back
    cumulative_p_delay = np.pad(
        cumulative_p_delay,
        padding_shape,
        mode='constant',
        constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)

    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay

    return adjusted, cumulative_p_delay

def plot_adjusted_data(region,confirmed,onset,adjusted):
    ###Take a look at all three series: confirmed, onset and onset adjusted
    # for right censoring.

    fig, ax = plt.subplots(figsize=(5,3))

    confirmed.plot(
        ax=ax,
        label='Confirmed',
        title=region,
        c='k',
        alpha=.25,
        lw=1)

    onset.plot(
        ax=ax,
        label='Onset',
        c='k',
        lw=1)

    adjusted.plot(
        ax=ax,
        label='Adjusted Onset',
        c='k',
        linestyle='--',
        lw=1)

    ax.legend();



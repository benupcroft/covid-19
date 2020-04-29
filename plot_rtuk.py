"""### Render Charts"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import shutil
import time

def plot_rt():
    results = pd.read_csv('data/latest_results.csv',
                          parse_dates=['date'],
                          index_col=['region', 'date']).sort_index()

    c = (0.3, 0.3, 0.3, 1)
    ci = (0, 0, 0, 0.05)

    ncols = 3
    nrows = int(np.ceil(results.index.levels[0].shape[0] / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, nrows * 3),
        sharey='row')

    for ax, (region, result) in zip(axes.flat, results.groupby('region')):

        result = result.droplevel(0)

        ax.set_ylim(0.5, 1.6)
        ax.set_title(region)
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

    fig.tight_layout()
    fig.set_facecolor('w')

    # plt.show()

    # move the old plot and save
    timestr = time.strftime("%Y%m%d-%H%M%S")
    shutil.move("web/static/img/latest.png", "web/static/img/old_plots/rt_plot-"+timestr+".png")
    fig.savefig('web/static/img/latest.png', dpi=fig.dpi)


# plot_rt()
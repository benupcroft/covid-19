#!/usr/bin/python3.7
"""Render Charts"""
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import shutil
import time


def plot_rt():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_directory, 'data/latest_results.csv')
    results = pd.read_csv(results_file,
                          parse_dates=['date'],
                          index_col=['region', 'date']).sort_index()

    c = (0.3, 0.3, 0.3, 1)
    ci = (0, 0, 0, 0.05)

    fig = plt.figure()
    ax = plt.gca()

    for indx, (region, result) in enumerate(results.groupby('region')):

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
        # ax.axvline(x=mdates.datetime.date(2020, 3, 12), linewidth=1, linestyle='--', color='tab:gray')
        # ax.text(mdates.datetime.date(2020, 3, 13), 0.6, 'Self-isolation', fontsize=8, rotation=90, color='tab:gray')
        # ax.axvline(x=mdates.datetime.date(2020,3,23), linewidth=1, linestyle='--', color='tab:orange')
        # ax.text(mdates.datetime.date(2020, 3, 24), 0.6, 'Nationwide lockdown', fontsize=8, rotation=90, color='tab:orange')
        # ax.axvline(x=mdates.datetime.date(2020,5,13), linewidth=1, linestyle='--', color='tab:orange')
        # ax.text(mdates.datetime.date(2020, 5, 12), 0.6, 'England - Return to work', fontsize=8, rotation=90, color='tab:orange')
        # ax.axvline(x=mdates.datetime.date(2020,5,28), linewidth=1, linestyle='--', color='tab:orange')
        # ax.text(mdates.datetime.date(2020, 5, 27), 0.6, 'Scotland - Phase 1 for easing lockdown', fontsize=8, rotation=90, color='tab:orange')
        # ax.axvline(x=mdates.datetime.date(2020,6,1), linewidth=1, linestyle='--', color='tab:orange')
        # ax.text(mdates.datetime.date(2020, 5,31), 0.6, 'England - Some year levels return to school', fontsize=8, rotation=90, color='tab:orange')
        ax.axvline(x=mdates.datetime.date(2020,6,18), linewidth=1, linestyle='--', color='tab:orange')
        ax.text(mdates.datetime.date(2020, 6,17), 0.6, 'Scotland - Phase 2 for easing lockdown', fontsize=8, rotation=90, color='tab:orange')
        ax.axvline(x=mdates.datetime.date(2020,6,22), linewidth=1, linestyle='--', color='tab:orange')
        ax.text(mdates.datetime.date(2020, 6,21), 0.6, 'Wales - Non-essential shops reopen', fontsize=8, rotation=90, color='tab:orange')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

        ax.set_ylabel(r'$R_t$', rotation='horizontal')
        ax.yaxis.set_label_coords(-0.05, 1.02)

        fig.tight_layout()
        fig.set_facecolor('w')

        # move the old plots and save the latest ones
        timestr = time.strftime("%Y%m%d-%H%M%S")

        fig_name = 'latest' + str(indx) + '.png'

        script_directory = os.path.dirname(os.path.abspath(__file__))
        web_directory = os.path.join(script_directory, '../../', 'rtuk/mysite')
        img_file = os.path.join(web_directory, 'static/img/' + fig_name)
        saved_img_file = os.path.join(web_directory, 'static/img/old_plots/rt_plot-' + str(indx) + '-' + timestr + '.png')
        # print('web_directory = ', web_directory)
        # print('img_file = ', img_file)
        # print('saved_img_file = ', saved_img_file)

        if os.path.exists(img_file):
            shutil.move(img_file, saved_img_file)

        fig.savefig(img_file, dpi=fig.dpi)

        plt.cla()

    # Write out the date and time to a json file and save in website directory
    # This file is used to display the last updated time and date on the website
    last_updated_time = time.strftime("%H:%M %d/%m/%Y")
    last_updated_time_file = os.path.join(web_directory, 'static/last_updated_time.json')

    with open(last_updated_time_file, "w") as text_file:
        text_file.write('{\n'
                        '\t\"last_updated_timestamp\": "' + last_updated_time + '"'
                        '\n}')

    # plt.show()


if __name__ == '__main__':
    plot_rt()


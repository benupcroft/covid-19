# RtUK - Covid-19

The code in this repository models the effective infection rate of Covid-19 
given past UK data over the last ~50 days and presents daily updates of that rate.

As the analysis uses past data to inform the current data, it estimates the "effective"
rate of infection of Covid-19. This is known as the Effective Reproduction Number Rt.

The results from this analysis are displayed daily on https://www.rtuk-live.com

The model for the analysis has been adapted from Systrom and Vladeck's model where they estimate Rt for US states. 
Systrom kindly released the model here: https://github.com/k-sys/covid-19/


## Quickstart
Run rt_live_uk.py: main script

This script will download latest data for the UK, create the model for computing Rt, and compute Rt for the last ~50 days. 


## Walkthrough description
The Jupyter files (*.ipynb) are a great walkthrough of how the model works using US data

Realtime R0.ipynb: in depth description

Realtime Rt mcmc.ipynb: includes the latest model using pymc3

region_comparison.py: Comparison of similar regions around the world at lockdown. 

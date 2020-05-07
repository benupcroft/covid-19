import pandas as pd
from prob_confirmation_delay import prob_delay
from load_data import load_lombardy_region_data, download_only_if_newer
from integrity_check import is_data_current, do_daily_cases_increase
from run_model import run_model

# Find the relationship between onset of symptoms and the delay until
# actual medical confirmation
# ~100mb download (be ... patient!)
p_delay = prob_delay()

lombardy_url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-regioni.json'

# is_updated, url_date = download_only_if_newer(lombardy_url)
# don't bother processing if data is still old
# if not is_updated:
# download and load up international regional data
regions1 = load_lombardy_region_data(lombardy_url)

# frames = [regions1, regions2, regions3, regions4]
frames = [regions1]
regions = pd.concat(frames)
# print(regions)

# Check the integrity of the data
is_data_current(regions)
do_daily_cases_increase(regions)

# covid_cases.loc[covid_cases['region'] == 'Lombardia']

run_model(p_delay, regions)

# Italy - Nationwide lockdown - 9 March (Lombardy region lockdown 8 March)

# Plot charts
# plot_rt()

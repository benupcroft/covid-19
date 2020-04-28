from datetime import date
from datetime import datetime

def is_data_current(regions):
    # # Make sure that all the regions have current data
    today = datetime.combine(date.today(), datetime.min.time())
    last_updated = regions.reset_index('Specimen date').groupby('Area name')['Specimen date'].max()
    is_current = last_updated < today

    try:
        assert is_current.sum() == 0
    except AssertionError:
        print("Not all regions have updated")
    #   print(last_updated[is_current])

def do_daily_cases_increase(regions):
    # Ensure all case diffs are greater than zero
    for region, grp in regions.groupby('Area name'):
        new_cases = grp['Cumulative lab-confirmed cases'].diff().dropna()
        is_positive = new_cases.ge(0)

        try:
            assert is_positive.all()
        except AssertionError:
            print(f"Warning: {region} has date with negative case counts")
    #        display(new_cases[~is_positive])


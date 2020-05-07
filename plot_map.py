import os
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib


# map information and borders
url_gb = 'https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/electoral/gb/eer.json'
# 'EER13CD', 'EER13CDO', 'EER13NM', 'geometry'
# 'E15000001,'01       , North East, wsg84 points
data_gb = requests.get(url_gb,
                       allow_redirects=True).text
gdata = gpd.read_file(data_gb)
# print(gdata.head())
# print(gdata.columns)
gdata.drop(['EER13CD','EER13CDO'],
           axis=1,
           inplace=True)

url_ni = 'https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/electoral/ni/eer.json'
data_ni = requests.get(url_ni,
                       allow_redirects=True).text
gdata_ni = gpd.read_file(data_ni)
# print(gdata_ni.head())
gdata_ni.drop(['ID', 'Area_SqKM', 'OBJECTID'],
              axis=1,
              inplace=True)

# rename gdata_ni.NAMES to EER13NM
# check how to concatenate

frames = [gdata, gdata_ni]
gdata_uk = pd.concat(frames)

script_directory = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(script_directory, 'data/latest_results.csv')
results = pd.read_csv(results_file,
                      parse_dates=['date'],
                      index_col=['region', 'date']).sort_index()


# print(results)
# print(results.index)

results = results.xs('2020-04-30',level='date')

results.drop(['mean',
             'lower_50',
             'upper_50',
             'lower_90',
             'upper_90'],
             axis=1,
             inplace=True)

# print(results)

# print(gdata)


gdata = gdata.join(results, on='EER13NM')

# Calculate centroids and plot
gdata_centroids = gdata.geometry.centroid
# Need to provide "zorder" to ensure the points are plotted above the polygons
fig = plt.figure()
ax = plt.gca()
gdata.plot(ax=ax,
           color='lightgrey',
           # column='median',
           edgecolor="white",
           linewidth=0.25)

# cmap = 'OrRd',

# gdata_centroids.plot(ax=ax,
#                      markersize=5,
#                      color='green',
#                      zorder=10)

# print(gdata_centroids)
print(gdata_centroids.x)
print(results['median'].values[0:10])


plt.scatter(gdata_centroids.x,
            gdata_centroids.y,
            s=2**(results['median'].values[0:11]*7),
            # color="green",
            cmap='OrRd',
            alpha=0.5)


plt.show()

# covid_cases = pd.read_csv(data_file,
#                           parse_dates=['Date'],
#                           index_col=['Date']).sort_index()

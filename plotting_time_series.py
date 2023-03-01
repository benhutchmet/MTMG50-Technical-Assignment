# %%
# import the relevant modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime
from datetime import date
from datetime import timedelta


# import the data
f = Dataset(r'C:\Users\Ben Hutchins\OneDrive - University of Reading\Documents\GitHub\MTMG50-Technical-Assignment\japan_ERA5land_20170601-20190801_tokuyama.nc', 'r')
# import the runoff data
runoff = f.variables['ro'][:,0,0]
# import the time data
time = f.variables['time'][:]
# convert the time data to a datetime format
# the time data is in hours since 1900-01-01 00:00:00
# so we need to add this to the start date
start = datetime.datetime(1900, 1, 1, 0, 0, 0)
newcal = [start + datetime.timedelta(hours = i) for i in time]
# check the time data


#print(newcal)


# create a pandas dataframe with the time data and runoff data
df = pd.DataFrame({'time': newcal, 'runoff': runoff})
# rename runoff to runoff (m)
df.rename(columns={'runoff': 'runoff (m)'}, inplace=True)
# check the dataframe
df


# plot the data
#plt.plot(df['time'], df['runoff (m)'])
#plt.show()


# create a function which will plot the data for any given start and end date
def plot_data(start, end):
    """Function which plots a time series of runoff at the tokuyama station for a given start and end date.

    Args:
        start (str): The start date in the format 'YYYY-MM-DD'.
        end (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
        None.
    """

    condition = df['time'].between(start, end)
    timeframe_code = df.loc[condition]
    plt.plot(timeframe_code['time'], timeframe_code['runoff (m)'])
    plt.show()


# test the function with the 2017 data
plot_data('2017-06-01', '2017-09-30')

# test the function with the 2018 data
plot_data('2018-06-01', '2018-09-30')

# %%

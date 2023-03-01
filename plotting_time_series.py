# %%
# import the relevant modules

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import datetime
from datetime import date
from datetime import timedelta
# %%

# import the data
f = Dataset(r"C:\Users\Ben Hutchins\OneDrive - University of "
            r"Reading\Documents\GitHub\MTMG50-Technical-Assignment\japan_ERA5land_20170601-20190801_tokuyama.nc", 'r')
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


print(newcal)

# %%

# create a pandas dataframe with the time data and runoff data
df = pd.DataFrame({'time': newcal, 'runoff': runoff})
# rename runoff to runoff (m)
df.rename(columns={'runoff': 'runoff (m)'}, inplace=True)
# check the dataframe
df

# %%

# plot the data
plt.plot(df['time'], df['runoff (m)'])
plt.show()

# %%
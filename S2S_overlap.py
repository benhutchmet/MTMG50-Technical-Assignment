
# %%
import xarray as xr


# format this as a function
# def S2S_data(filename, lat1, lat2, lon1, lon2):
#     """Load the data from a NetCDF file and select the data within the bounding box.
    
#     Inputs:
#     filename: the name of the NetCDF file
#     lat1: the minimum latitude
#     lat2: the maximum latitude
#     lon1: the minimum longitude
#     lon2: the maximum longitude

#     Outputs:
#         """

#     # load the data
#     data = xr.open_dataset(filename)
#     # group the data by the time coordinate and take the mean
#     averaged_data = data.groupby('time').mean(dim='time')
#     # save the averaged data to a new NetCDF file
#     averaged_data.to_netcdf(filename[:-3] + '_overlap_mean.nc')

# # call the function
# S2S_data(r'japan_ECMF_CF.20170601-20170915.nc', 35.167, 36.167, 136.002, 137.002)

# # for 2018
# S2S_data(r'japan_ECMF_CF.20180604-20180914.nc', 35.167, 36.167, 136.002, 137.002)

# # for 2019
# S2S_data(r'japan_ECMF_CF.20190603-20190913.nc', 35.167, 36.167, 136.002, 137.002)


# define a function to import the S2S data
def S2S_data(filename):
    """Load the data from a NetCDF file, converts the units and removes negative values. Also groups the data by the time coordinate and takes the mean."""

    # load the data
    data = xr.open_dataset(filename)

    # replace all values less than 0 with 0
    data['ro'] = data['ro'].where(data['ro'] > 0, 0)

    # convert the runoff to the same units as the ERA5 data
    # we will do this by dividing by 1000
    data['ro'] = data['ro']/1000

    # group the data by the time coordinate and take the mean
    averaged_data = data.groupby('time').mean(dim='time')

    # save the averaged data to a new NetCDF file
    averaged_data.to_netcdf(filename[:-3] + '_overlap_mean.nc')

# call the function
S2S_data(r'japan_ECMF_PF.diff.study_area.nc')

# check the data
data = xr.open_dataset(r'japan_ECMF_PF.diff.study_area_overlap_mean.nc')

# plot the runoff data
#data['ro'].plot()


# for ERA5 data
# we need to import the coarse resolution data
# for the full field
# this was generated using bilinear interpolation in CDO

era5 = xr.open_dataset(r'C:\Users\Ben Hutchins\OneDrive - University of Reading\Documents\GitHub\MTMG50-Technical-Assignment\japan_ERA5land_20170601-20190801_coarse_subset.nc')

# plot a histogram of the runoff data
#era5['ro'].plot.hist()
# what are the minimum and maximum values?
# print(era5['ro'].min().values)
# print(era5['ro'].max().values)

# # plot the runoff data
#era5['ro'].plot()

# # plot the s2s data
#data['ro'].plot()


# # now we want to plot the two probability density functions
# # on the same plot
# # we will use the seaborn library for this
import seaborn as sns
import matplotlib.pyplot as plt

# plot the two distributions on the same plot
# sns.distplot(data['ro'].values, label='S2S')
# sns.distplot(era5['ro'].values, label='ERA5')
# # set the limits of the x-axis
# plt.xlim(-0.02, 0.02)

# plot the two distributions on the same plot
sns.distplot(data['ro'].values.flatten(), label='S2S')
sns.distplot(era5['ro'].values.flatten(), label='ERA5')
plt.xlabel('Runoff (mm/day)')
# plot the means as vertical lines
plt.axvline(data['ro'].mean().values, color='blue')
plt.axvline(era5['ro'].mean().values, color='orange')
# include the standard deviations
plt.axvline(data['ro'].mean().values + data['ro'].std().values, color='blue', linestyle='--')
plt.axvline(data['ro'].mean().values - data['ro'].std().values, color='blue', linestyle='--')
plt.axvline(era5['ro'].mean().values + era5['ro'].std().values, color='orange', linestyle='--')
plt.axvline(era5['ro'].mean().values - era5['ro'].std().values, color='orange', linestyle='--')
# constrain the x-axis
plt.xlim(-0.02, 0.02)
plt.legend()
plt.show()
# save the plot in the plots folder
plt.savefig(r'plots\runoff_distribution_ERA5vsECMWFS2S_coarseERA5.png', dpi=300)

# %%

# for calibration purposes, we need to plot the ERA5 runoff agfainst the S2S runoff
# ERA5 runoff on the x-axis
# S2S runoff on the y-axis
# but first we need to get the S2S runoff in the same format as the ERA5 runoff

# to do this we need to group the data by the time coordinate
# first we will constrain to the 2017 period
# from 2017-06-01 to 2017-09-15
data_2017 = data.sel(time=slice('2017-06-01', '2017-09-15'))

era5_2017 = era5.sel(time=slice('2017-06-01', '2017-09-15'))
# now we need to group the data by the time coordinate to removce duplicates values in the ERA5 data
era5_2017 = era5_2017.groupby('time').mean(dim='time')


# print the dimensions of the data
print(data_2017.dims)
print(era5_2017.dims)

# average the data over the ensemble members
data_2017 = data_2017.mean(dim='number')

# plot the data on a scatter plot
# THIS IS SEPERATE FROM THE PLT.PLOT COMMANDS ABOVE

# set the figure size
plt.figure(figsize=(6,6))
# plot the data, with different colours for era5 and s2s
plt.scatter(era5_2017['ro'].values.flatten(), data_2017['ro'].values.flatten(), color='blue', label='ERA5')
# plot the 1:1 line
plt.plot([0, 0.01], [0, 0.01], color='black', linestyle='--')
# set the axis labels
plt.xlabel('ERA5 runoff (mm/day)')
plt.ylabel('S2S runoff (mm/day)')
# show the plot
plt.show()
# save the plot in the plots folder
plt.savefig(r'plots\runoff_scatter_ERA5vsECMWFS2S_coarseERA5.png', dpi=300)

# we want to perform a statistical calibration of the data
# first we must calculate the mean and standard deviation of the S2S data and the ERA5 data
# we will use the numpy library for this
import numpy as np

# calculate the mean and standard deviation of the S2S data
s2s_mean = data_2017['ro'].mean().values
s2s_std = data_2017['ro'].std().values

# calculate the mean and standard deviation of the ERA5 data
era5_mean = era5_2017['ro'].mean().values
era5_std = era5_2017['ro'].std().values


# now we need to calculate the value of C_0 and C_1
c0 = era5_mean - (s2s_mean * (era5_std/s2s_std))
c1 = era5_std/s2s_std

# now we need to apply the calibration to the S2S data
# we will do this by multiplying the S2S data by C_1 and adding C_0

# first we need to create a new variable in the data
data_2017['ro_calibrated'] = data_2017['ro'].values.flatten() * c1

print('c1 is:',c1)
print(c0)

# print ro and ro_calibrated to check that the calibration has worked
print(data_2017['ro_calibrated'].values.flatten())

# now we want to plot the probability density functions of the calibrated data
# and the ERA5 data
# on the same plot
# we will use the seaborn library for this
import seaborn as sns
import matplotlib.pyplot as plt

# plot the two distributions on the same plot with era5 in orange and s2s in purple
sns.distplot(data_2017['ro_calibrated'].values.flatten(), label='S2S calibrated')
sns.distplot(era5_2017['ro'].values.flatten(), label='ERA5')
plt.xlabel('Runoff (mm/day)')
# plot the means as vertical lines
plt.axvline(data_2017['ro_calibrated'].mean().values, color='blue')
plt.axvline(era5_2017['ro'].mean().values, color='orange')
# include the standard deviations
plt.axvline(data_2017['ro_calibrated'].mean().values + data_2017['ro_calibrated'].std().values, color='blue', linestyle='--')
plt.axvline(data_2017['ro_calibrated'].mean().values - data_2017['ro_calibrated'].std().values, color='blue', linestyle='--')
plt.axvline(era5_2017['ro'].mean().values + era5_2017['ro'].std().values, color='orange', linestyle='--')
plt.axvline(era5_2017['ro'].mean().values - era5_2017['ro'].std().values, color='orange', linestyle='--')
# constrain the x-axis
plt.xlim(-0.05, 0.05)
plt.legend()
plt.savefig(r'plots\runoff_distribution_ERA5vsECMWFS2S_calibrated.png', dpi=300)
plt.show()


# plot the data on a scatter plot
# showing both the uncailbrated and calibrated data

# set the figure size
plt.figure(figsize=(6,6))
# plot the data, with different colours for era5 and s2s
plt.scatter(era5_2017['ro'].values.flatten(), data_2017['ro'].values.flatten(), color='blue', label='uncalibrated')
plt.scatter(era5_2017['ro'].values.flatten(), data_2017['ro_calibrated'].values.flatten(), color='red', label='S2S calibrated')
# plot the 1:1 line
plt.plot([0, 0.02], [0, 0.02], color='black', linestyle='--')
# set the axis labels
plt.xlabel('ERA5 runoff (mm/day)')
plt.ylabel('S2S runoff (mm/day)')
# set x and y limits
plt.xlim(0, 0.02)
plt.ylim(0, 0.02)
# show the plot
plt.show()

# save the plot in the plots folder
plt.savefig(r'plots\runoff_scatter_ERA5vsECMWFS2S_calibrated.png', dpi=300)


# %%
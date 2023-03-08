
# %%
import xarray as xr

# format this as a function
def S2S_data(filename, lat1, lat2, lon1, lon2):
    """Load the data from a NetCDF file and select the data within the bounding box.
    
    Inputs:
    filename: the name of the NetCDF file
    lat1: the minimum latitude
    lat2: the maximum latitude
    lon1: the minimum longitude
    lon2: the maximum longitude

    Outputs:
        """

    # load the data
    data = xr.open_dataset(filename)
    # group the data by the time coordinate and take the mean
    averaged_data = data.groupby('time').mean(dim='time')
    # save the averaged data to a new NetCDF file
    averaged_data.to_netcdf(filename[:-3] + '_overlap_mean.nc')

# call the function
S2S_data(r'japan_ECMF_CF.20170601-20170915.nc', 35.167, 36.167, 136.002, 137.002)

# for 2018
S2S_data(r'japan_ECMF_CF.20180604-20180914.nc', 35.167, 36.167, 136.002, 137.002)

# for 2019
S2S_data(r'japan_ECMF_CF.20190603-20190913.nc', 35.167, 36.167, 136.002, 137.002)

# test to see how xarray handles ensemble members
data = xr.open_dataset(r'japan_ECMF_PF.20190627.nc')

# plot the runoff data
data['msl'].plot()

data.info()
# %%

# using cdo I have created a time series of data for the run off
# from 2017-06-01 to 2019-08-01

# I want to extract the data for the following coordinates:
# lon = 136.47642
# lat = 35.63422

# to do this I will use bilinear interpolation
# with the following cdo command:

cdo remapbil,lon=136.47642/lat=35.63422 japan_ERA5land_20170601-20190801.nc japan_ERA5land_20170601-20190801_tokuyama.nc


# command for extracting the data for the ecmwf data
# we only want to extract the RO variable
# and we want to use the CF foreacsts

# this will be for the 2017 season
cdo daymean -mergetime -select,name=ro *CF.2017*.nc japan_ECMF_CF.20170601-20170915.nc
cdo daymean -mergetime -select,name=ro *CF.2018*.nc japan_ECMF_CF.20180604-20180914.nc
cdo daymean -mergetime -select,name=ro *CF.2019*.nc japan_ECMF_CF.20190603-20190913.nc

# because of how the sesonal forecasts are set up, tehre are duplicates for each day
# we will just take the mean to get the data for the day
# there are limitations in doing this however, which will be discussed in the report


# select the lon lat box and compute a mean from this
cdo fldmean -sellonlatbox,136.002,137.002,35.167,36.167 japan_ECMF_CF.20170601-20190913.nc outfile.nc
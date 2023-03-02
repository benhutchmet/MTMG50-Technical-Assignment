# using cdo I have created a time series of data for the run off
# from 2017-06-01 to 2019-08-01

# I want to extract the data for the following coordinates:
# lon = 136.47642
# lat = 35.63422

# to do this I will use bilinear interpolation
# with the following cdo command:

cdo remapbil,lon=136.47642/lat=35.63422 japan_ERA5land_20170601-20190801.nc japan_ERA5land_20170601-20190801_tokuyama.nc

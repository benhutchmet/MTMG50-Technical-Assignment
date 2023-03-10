#!/bin/bash

# this script calculate the runoff for each of the PF. files
# for each file, we want to:
# 1. extract the runoff data ['ro']
# 2. we want to calculate the difference between the runoff at the current time step and the runoff at the previous time step for each ensemble member
# 3. we want to calculate the mean over all of the ensemble members
# this should only give positive values for the runoff data

# first we are going to write a test script to perform the above steps for one file

# we will use the following file as a test
# japan_ECMF_PF.20170601.nc

# the ensemble members are stored as the following variable:
# 	double number(number) ;
#		number:long_name = "ensemble_member" ;
#		number:axis = "Z" ;

# the runoff data is stored as the following variable:
#	short ro(time, number, latitude, longitude) ;
#		ro:long_name = "Water runoff and drainage" ;
#		ro:units = "kg m**-2" ;
#		ro:add_offset = 1211.1065188531 ;
#		ro:scale_factor = 0.036962293806174 ;
#		ro:_FillValue = -32767s ;
#		ro:missing_value = -32767s ;

# initialise the for loop to loop over all of the ensemble members
for i in {1..50}; do

    # extract the runoff data for the current ensemble member
    cdo selname,ro -sellevel,$i japan_ECMF_PF.20170601.nc japan_ECMF_PF.20170601.$i.nc

    # calculate the difference between the runoff at the current time step and the runoff at the previous time step within the file
    cdo -b f64 deltat japan_ECMF_PF.20170601.$i.nc japan_ECMF_PF.20170601.$i.diff.nc

    # remove the temporary files
    rm japan_ECMF_PF.20170601.$i.nc

done

# calculate the mean over all of the ensemble members
cdo ensmean japan_ECMF_PF.20170601.*.diff.nc japan_ECMF_PF.20170601.diff.nc

# now we want to write a script to perform the above steps for all of the PF files
# the PF files are given in the following format:


# we want to loop over all of the PF files:
for i in japan_ECMF_PF.????????.nc; do

    # extract the date from the file name
    date=$(echo $i | cut -d'.' -f2)

    # initialise the for loop to loop over all of the ensemble members
    for j in {1..50}; do

        # extract the runoff data for the current ensemble member
        cdo selname,ro -sellevel,$j $i japan_ECMF_PF.$date.$j.nc

        # calculate the difference between the runoff at the current time step and the runoff at the previous time step within the file
        cdo -b f64 deltat japan_ECMF_PF.$date.$j.nc japan_ECMF_PF.$date.$j.diff.nc

        # remove the temporary files
        rm japan_ECMF_PF.$date.$j.nc

    done

    # calculate the mean over all of the ensemble members
    cdo ensmean japan_ECMF_PF.$date.*.diff.nc japan_ECMF_PF.$date.diff.nc

    # remove the temporary files
    rm japan_ECMF_PF.$date.*.diff.nc

done


# combine all of the PF diff files into one file
cdo mergetime japan_ECMF_PF.*.diff.nc japan_ECMF_PF.diff.nc

# select the study area
cdo sellonlatbox,135,137,34,36 japan_ECMF_PF.diff.nc japan_ECMF_PF.diff.study_area.nc

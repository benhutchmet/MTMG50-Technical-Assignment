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


# for task Q5 we want to run the dam model over the full duration
# of one S2S forecast starting in June 2018
# for this we will use the 20180621 data
# japan_ECMF_PF.20180621.nc

# first we will calculate the runoff for the 20180621 data
for i in {1..50}; do

    # extract the runoff data for the current ensemble member
    cdo selname,ro -sellevel,$i japan_ECMF_PF.20180621.nc japan_ECMF_PF.20180621.$i.nc

    # calculate the difference between the runoff at the current time step and the runoff at the previous time step within the file
    # put the output file in the 20180621 folder
    cdo -b f64 deltat japan_ECMF_PF.20180621.$i.nc /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.nc

    # divide the runoff by 1000 to convert from kg m-2 to m3 s-1
    cdo divc,1000 /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.nc /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.nc

    # remove the temporary files
    rm japan_ECMF_PF.20180621.$i.nc
    rm /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.nc

done

# we then need to select the lat/lon box for the study area for each of the files
for i in {1..50}; do

    # select the lat/lon box for the study area
    cdo sellonlatbox,135,137,34,36 /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.nc /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.study_area.nc

    # remove the temporary files
    rm /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.nc

done

# we then want to calculate the mean for the grid box for each of the ensemble members
for i in {1..50}; do

    # calculate the mean for the grid box
    cdo fldmean /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.study_area.nc /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.study_area.mean.nc

    # remove the temporary files
    rm /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.study_area.nc

done

# c1 is: 2.057371428376461

# we want to calibrate the data using the c1 value
# this acts to scale the data to the observed data in ERA5

# define the c1 value
c1=2.057371428376461

# we then want to calculate the runoff for the 20180621 data
for i in {1..50}; do

    # perform the calibration
    cdo mulc,$c1 /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.study_area.mean.nc /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.study_area.mean.calibrated.nc

    # remove the temporary files
    rm /home/users/benhutch/MTMG50/S2S/ecmwf/20180621/japan_ECMF_PF.20180621.$i.diff.converted-units.study_area.mean.nc

done



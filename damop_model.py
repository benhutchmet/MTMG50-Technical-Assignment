"""
Applying an impact model for hydroelectric dam management driven by
a time series of runoff data

Author: 2020, John Methven
Revised to include relief flow and smoother solution
        2021, John Methven
"""
# %%
# Library functions needed to run damop_model()
# Not including standard imports like numpy
#
from scipy import optimize
from scipy import signal
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import datetime
from datetime import date
from datetime import timedelta

# import from dictionary
from dictionaries import *

# check the data
#print(params_Q1a)




def damop_model(runoffarr, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma):
    '''
    Implementation of the dam operation model of Hirsch et al (2014).
    Input: 
    :runoffarr  - input time series for runoff data
    :dt         - runoff accumulation interval per record
    :catcharea  - catchment area for the dam
    :kappa      - parameter relating reservoir depth to volume
    :hmax       - maximum water head (constraint on optimization)
    :hmin       - minimum water head
    :wmax       - maximum flow rate through turbines
    :wmin       - minimum flow rate to maintain some power generation
    :rmax       - maximum relief flow rate, bypassing turbines in flood
    :sigma      - operational efficiency of power generation by dam
    Output: 
    :inflow     - input time series for inflow to reservoir  
    :x          - output time series for water head at dam
    :w          - output solution for optimum flow rate through turbines
    :r          - output solution for relief flow rate
    :gout       - value of time integrated generation for optimum solution (MW-days)
    '''       
    print()
    print('damop_model has been called with the constraints:')
    print('wmax = ',wmax,'   wmin = ',wmin,'   hmax = ',hmax,'   hmin = ',hmin)
    #
    # Convert runoff data from units of m to an equivalent inflow in m^3 s^-1
    # Assume that the same runoff rate applies to the entire catchment area for dam
    #
    runoffave = np.mean(runoffarr)
    inflow = catcharea*runoffarr/dt
    n = len(inflow)
    inmax = max(inflow)
    #
    # Set parameter used to control computational mode using filter similar to Robert-Asselin
    # Recommend 0 because filter introduces an offset of W relative to I in optimization.
    #
    alpha = 0.0
    #
    # Apply running mean to the inflow data if required for smoother solution 
    # to the optimisation. Averaging window length = nwin.
    #
    nwin = 3
    inflow = running_mean(inflow, nwin)
    #
    # Scale mu so that the sum of generation over time points is approx one.
    # This gives a better numerical solution in the optimisation for max generation
    # by reducing numerical truncation error in the calculation.
    #
    mu = 1.0/(n*sigma*wmax*hmax)
    #
    # The dam management optimization model is set up in the mathematical form of a 
    # quadratic programming problem.
    # The only input time series is the inflow to the reservoir.
    # The model solves for the water head at the dam maximizing power generation.
    # This then gives the flow rate through the turbines.
    # However, contraints are applied on maximum and minimum water level 
    # and maximum/minimum flow rate through the turbines.
    #
    # The equation for generation can be written in the form
    # 
    # G = 0.5*H^T P H + q^T H
    #
    # where H is the head time series we are solving for (a 1-D array) and 
    # P is a matrix and q is also a 1-D time series (scaled inflow).
    # The notation ^T means the transpose of the matrix. 
    # Quadratic programming aims to minimize -G which is equivalent to max(G).
    #
    q = -mu*sigma*inflow
    umat = np.zeros((n, n))
    inmat = np.zeros((n, n))
    cmat = np.zeros((n, n))
    for i in range(n):
        umat[i, i] = 1
        inmat[i, i] = inflow[i]

    for j in range(n-2):
        i = j+1
        cmat[i, i-1] = -1 + 0.5*alpha
        cmat[i, i]   = -alpha
        cmat[i, i+1] = 1 + 0.5*alpha
    
    pscal = mu*sigma*(kappa/dt)*cmat
    wscal = -0.5*(kappa/dt)*cmat
    #
    # Set constraints on the water head at the dam: hmin <= h <= hmax
    # Optimization requires that constraints actually need to be applied in form:
    # Amat x <= b  (where in this problem the vector x is head time series, h).
    # For Amat x >= b it is necessary to re-arrange to -Amat x <= -b.
    # Therefore to apply hmin <= h <= hmax, the matrix Amat is the unit matrix.
    #
    hscal = umat
    hmaxcons = np.ones(n)*hmax
    hmincons = np.ones(n)*hmin    
    #
    # Set constraints on the flow rate 
    # based on the parameters Wmax, Rmax and Wmin.
    # The form of the contraints means that it must be applied to range of W*h:
    # Wmin*hmin <= W*h <= (Wmax+Rmax)*hmax
    #
    gscal = wscal + inmat
    gmaxcons = np.zeros(n)
    gmincons = np.zeros(n)
    for i in range(n):
        gmaxcons[i] = (wmax+rmax)*hmax
        gmincons[i] = wmin*0.5*(hmin+hmax)
    #
    # Construct a single matrix describing Amat and vector for constraint values b
    # in the form required by optimize.minimize
    #
    vmat = np.concatenate((gscal, -gscal, hscal, -hscal), axis=0)
    vcons = np.concatenate((gmaxcons, -gmincons, hmaxcons, -hmincons))
    
    print('Now apply quadratic minimization technique')
    
    def gen(x, sign=1.):
        return sign * (0.5*np.dot(x.T, np.dot(pscal, x)) + np.dot(q.T, x))
    
    def jac(x, sign=1.):
        return sign * (np.dot(x.T, pscal) + q.T)
    
    cons = {'type':'ineq',
            'fun':lambda x: vcons - np.dot(vmat, x),
            'jac':lambda x: -vmat}
    
    opt = {'disp':True, 'maxiter':100, 'ftol':1e-08}

    #
    # Obtain solution by minimization nouter times. Smooth the input first guess 
    # and results for head, h, which removes noise and any numerical instability in 
    # optimal solution for the flow rate time series, W.
    # Note that the minimize method does not always find a solution consistent 
    # with the contraints imposed (depending on the first guess data) and these
    # failed attempts are not included in the average solution.
    #
    nouter = 3
    istsuccess = 1
    ic = -1
    afac = 0.5
    xinit = hmax*(afac + 0.1*np.random.randn(n))
    nwin = min([41, 2*round(0.2*n)+1])
    print('running mean window length, nwin = ',nwin)
    xinit = running_mean(xinit, nwin)
    
    for io in range(nouter):
    #while istsuccess == 1:
        #
        # First guess values for x (water head).
        # Random variation on top of constant level.
        # Smooth to reduce 2-grid noise in input data.
        #
        ic = ic+1
        res_cons = optimize.minimize(gen, xinit, jac=jac, constraints=cons,
                                 method='SLSQP', options=opt)
        xup = res_cons['x']
        fup = res_cons['fun']  
        stexit = res_cons['status']
    
        if stexit != 4:
            if istsuccess == 1:
                x = xup
                x = running_mean(x, nwin)
                xinit = x
                f = fup
                print('Constrained optimization')
                print(res_cons)
                print('iter ',ic,' f = ',f)
                istsuccess = 0
            else:
                if (fup/f) < 2:
                    afac = float(ic+1)/nouter
                    x = afac*x + (1-afac)*xup
                    x = running_mean(x, nwin)
                    xinit = x
                    f = afac*f + (1-afac)*fup
                    print('iter ',ic,' f = ',f)
        if ic == nouter:
            print(nouter,' outer iterations finished without reaching result')
            istsuccess = 1
    # end outer loop
    
    #
    # Optimisation returns the head in variable x
    # Total flow rate ft = W+R is calculated from head and known inflow rate
    # Total flow is diverted into relief flow when it exceeds Wmax (and Rmax > 0)
    #
    ft = np.dot(wscal, x) + inflow
    w = np.copy(ft)
    r = np.zeros(n)
    excessflow = np.where(ft > wmax)
    if rmax > 0:
        w[excessflow] = wmax
        r[excessflow] = ft[excessflow]-wmax
    
    gout = -f
    
    return inflow, x, w, r, gout


def running_mean(xarr, nwin):
    '''
    Apply running mean filter through array
    Inputs:
        xarr    - array to filter
        nwin    - number of points in the filter window (odd number expected)
    Output:
        xfilt   - same length as xarr after application of filter
    '''
    n = len(xarr)
    xfilt = np.copy(xarr)
    ist = int(nwin/2)
    xconv = np.convolve(xarr, np.ones(nwin),'valid')/nwin
    nconv = len(xconv)
    xfilt[ist:n-ist] = xconv[:]
    xfilt[0:ist] = xconv[0]
    xfilt[n-ist:n] = xconv[nconv-1]
    
    return xfilt



# define an updated version of the damop model

def damop_model_UPDATED(params):
    '''
    Implementation of the dam operation model of Hirsch et al (2014).
    
    Inputs:
    params  - list of parameters for the model

    Outputs:
    inflow   - inflow rate to the dam (m^3 s^-1)
    x - head in the reservoir (m)
    w - flow rate through the dam (m^3 s^-1)
    r - flow rate through the relief channel (m^3 s^-1)
    gout - objective function value (negative of the total flow rate)                     
    '''
    # first import the constants from the dictionary
    catcharea = params['catchment_area'] # the area of the catchment (m^2)
    kappa = params['kappa'] # proportionality constant between resv. volume and head
    hmax = params['H_max'] # maximum safe water height in the reservoir (m)
    hmin = params['H_min'] # minimum safe water height in the reservoir (m)
    wmax = params['W_max'] # maximum safe flow rate through the dam (m^3 s^-1)
    wmin = params['W_min'] # minimum flow rate through the dam (m^3 s^-1)
    rmax = params['R_max'] # maximum flow rate through the relief channel avoiding turbines (m^3 s^-1)
    sigma = params['sigma'] # efficiency of power generation (proportion)

    # import the task flag and fig name
    task = params['task']
    fig_name = params['fig_name']

    # now we want to import the runoff data
    # first import the path for the netcdf point file
    path = params['path']
    # now import the data
    f = Dataset(path, 'r')
    # import runoff and time data
    # this part was updated with the help of james to make//
    # the model run faster lol
    # for some reason netcdf file is in four hourly chunks?gi
    # set the runoffarr and time arrays
    runoffarr = f.variables['ro'][:,0,0]
    time = f.variables['time'][:]
     # close the file
    f.close()

    # convert the time to a datetime object
    start = datetime.datetime(1900, 1, 1, 0, 0, 0) # hours since 1900-01-01 00:00:00
    time = [start + datetime.timedelta(hours=t) for t in time]

    # create a dataframe to store the data
    df = pd.DataFrame({'time':time, 'runoff':runoffarr})

    # we want to constrain the model to only run for a certain period of time
    # first we need to import the start and end dates
    # these should be in the format yyyy-mm-dd
    start_date = params['start_date']
    end_date = params['end_date']

    # define the condition which constrains the data
    condition = (df['time'].between(start_date, end_date))
    constrained_df = df.loc[condition]
    # now we want to set the constrained data as arrays
    runoffarr = constrained_df['runoff'].to_numpy()
    timearr = constrained_df['time'].to_numpy()

    # now we want to set the timestep for converting the runoff data below
    dt = params['dt'] # runoff accumulation interval per record (s)

    print()
    print('damop_model has been called with the constraints:')
    print('wmax = ',wmax,'   wmin = ',wmin,'   hmax = ',hmax,'   hmin = ',hmin)
    #
    # Convert runoff data from units of m to an equivalent inflow in m^3 s^-1
    # Assume that the same runoff rate applies to the entire catchment area for dam
    #
    runoffave = np.mean(runoffarr)
    inflow = catcharea*runoffarr/dt
    n = len(inflow)
    inmax = max(inflow)
    #
    # Set parameter used to control computational mode using filter similar to Robert-Asselin
    # Recommend 0 because filter introduces an offset of W relative to I in optimization.
    #
    alpha = 0.0
    #
    # Apply running mean to the inflow data if required for smoother solution 
    # to the optimisation. Averaging window length = nwin.
    #
    nwin = 3
    inflow = running_mean(inflow, nwin)
    #
    # Scale mu so that the sum of generation over time points is approx one.
    # This gives a better numerical solution in the optimisation for max generation
    # by reducing numerical truncation error in the calculation.
    #
    mu = 1.0/(n*sigma*wmax*hmax)
    #
    # The dam management optimization model is set up in the mathematical form of a 
    # quadratic programming problem.
    # The only input time series is the inflow to the reservoir.
    # The model solves for the water head at the dam maximizing power generation.
    # This then gives the flow rate through the turbines.
    # However, contraints are applied on maximum and minimum water level 
    # and maximum/minimum flow rate through the turbines.
    #
    # The equation for generation can be written in the form
    # 
    # G = 0.5*H^T P H + q^T H
    #
    # where H is the head time series we are solving for (a 1-D array) and 
    # P is a matrix and q is also a 1-D time series (scaled inflow).
    # The notation ^T means the transpose of the matrix. 
    # Quadratic programming aims to minimize -G which is equivalent to max(G).
    #
    q = -mu*sigma*inflow
    umat = np.zeros((n, n))
    inmat = np.zeros((n, n))
    cmat = np.zeros((n, n))
    for i in range(n):
        umat[i, i] = 1
        inmat[i, i] = inflow[i]

    for j in range(n-2):
        i = j+1
        cmat[i, i-1] = -1 + 0.5*alpha
        cmat[i, i]   = -alpha
        cmat[i, i+1] = 1 + 0.5*alpha
    
    pscal = mu*sigma*(kappa/dt)*cmat
    wscal = -0.5*(kappa/dt)*cmat
    #
    # Set constraints on the water head at the dam: hmin <= h <= hmax
    # Optimization requires that constraints actually need to be applied in form:
    # Amat x <= b  (where in this problem the vector x is head time series, h).
    # For Amat x >= b it is necessary to re-arrange to -Amat x <= -b.
    # Therefore to apply hmin <= h <= hmax, the matrix Amat is the unit matrix.
    #
    hscal = umat
    hmaxcons = np.ones(n)*hmax
    hmincons = np.ones(n)*hmin    
    #
    # Set constraints on the flow rate 
    # based on the parameters Wmax, Rmax and Wmin.
    # The form of the contraints means that it must be applied to range of W*h:
    # Wmin*hmin <= W*h <= (Wmax+Rmax)*hmax
    #
    gscal = wscal + inmat
    gmaxcons = np.zeros(n)
    gmincons = np.zeros(n)
    for i in range(n):
        gmaxcons[i] = (wmax+rmax)*hmax
        gmincons[i] = wmin*0.5*(hmin+hmax)
    #
    # Construct a single matrix describing Amat and vector for constraint values b
    # in the form required by optimize.minimize
    #
    vmat = np.concatenate((gscal, -gscal, hscal, -hscal), axis=0)
    vcons = np.concatenate((gmaxcons, -gmincons, hmaxcons, -hmincons))
    
    print('Now apply quadratic minimization technique')
    
    def gen(x, sign=1.):
        return sign * (0.5*np.dot(x.T, np.dot(pscal, x)) + np.dot(q.T, x))
    
    def jac(x, sign=1.):
        return sign * (np.dot(x.T, pscal) + q.T)
    
    cons = {'type':'ineq',
            'fun':lambda x: vcons - np.dot(vmat, x),
            'jac':lambda x: -vmat}
    
    opt = {'disp':True, 'maxiter':100, 'ftol':1e-08}

    #
    # Obtain solution by minimization nouter times. Smooth the input first guess 
    # and results for head, h, which removes noise and any numerical instability in 
    # optimal solution for the flow rate time series, W.
    # Note that the minimize method does not always find a solution consistent 
    # with the contraints imposed (depending on the first guess data) and these
    # failed attempts are not included in the average solution.
    #
    nouter = 3
    istsuccess = 1
    ic = -1
    afac = 0.5
    xinit = hmax*(afac + 0.1*np.random.randn(n))
    nwin = min([41, 2*round(0.2*n)+1])
    print('running mean window length, nwin = ',nwin)
    xinit = running_mean(xinit, nwin)
    
    for io in range(nouter):
    #while istsuccess == 1:
        #
        # First guess values for x (water head).
        # Random variation on top of constant level.
        # Smooth to reduce 2-grid noise in input data.
        #
        ic = ic+1
        res_cons = optimize.minimize(gen, xinit, jac=jac, constraints=cons,
                                 method='SLSQP', options=opt)
        xup = res_cons['x']
        fup = res_cons['fun']  
        stexit = res_cons['status']
    
        if stexit != 4:
            if istsuccess == 1:
                x = xup
                x = running_mean(x, nwin)
                xinit = x
                f = fup
                print('Constrained optimization')
                print(res_cons)
                print('iter ',ic,' f = ',f)
                istsuccess = 0
            else:
                if (fup/f) < 2:
                    afac = float(ic+1)/nouter
                    x = afac*x + (1-afac)*xup
                    x = running_mean(x, nwin)
                    xinit = x
                    f = afac*f + (1-afac)*fup
                    print('iter ',ic,' f = ',f)
        if ic == nouter:
            print(nouter,' outer iterations finished without reaching result')
            istsuccess = 1
    # end outer loop
    
    #
    # Optimisation returns the head in variable x
    # Total flow rate ft = W+R is calculated from head and known inflow rate
    # Total flow is diverted into relief flow when it exceeds Wmax (and Rmax > 0)
    #
    ft = np.dot(wscal, x) + inflow
    w = np.copy(ft)
    r = np.zeros(n)
    excessflow = np.where(ft > wmax)
    if rmax > 0:
        w[excessflow] = wmax
        r[excessflow] = ft[excessflow]-wmax
    
    gout = -f

    # set up the optional plotting scripts if task = Q1 or Q2
    if task == 'Q1' or task == 'Q2':
        dam_model(start_date, end_date, inflow, x, w, r, gout, fig_name)

    # plotting scripts

def dam_model(start_date, end_date, inflow, x, w, r, gout, fig_name):
    """Run the dam model and plot the results
    """
    #create a dataframe to store the results for inflow, x, w, and r
    df = pd.DataFrame({'inflow':inflow, 'x':x, 'w':w, 'r':r})
    #set up the time variable for plotting with daily intervals
    time = pd.date_range(start_date, end_date, freq='D')

    # print the characteristics of the dataframe
    print(df.describe())

    #add time to the dataframe, excluding the last value
    df['time'] = time[:-1]

    #plot the results
    #inflow, w and r are plotted on the same axis
    #x is plotted on a separate axis
    fig, ax1 = plt.subplots()
    # flow rate in m^3/s
    ax1.plot(df['time'], df['inflow'], color='blue', label = 'inflow (m' + r'$^3$' + 's' + r'$^{-1}$' + ')')
    ax1.plot(df['time'], df['w'], color='red', label='dam flow (m' + r'$^3$' + 's' + r'$^{-1}$' + ')' )
    ax1.plot(df['time'], df['r'], color='green', label='relief flow (m' + r'$^3$' + 's' + r'$^{-1}$' + ')')
    ax1.set_xlabel('time')
    ax1.set_ylabel('flow rate (m' + r'$^3$' + 's' + r'$^{-1}$' + ')')
    ax2 = ax1.twinx()
    # set the strings on the x axis to curve
    for label in ax1.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')
    # set the opacity of the reservoir head plot to 0.5
    ax2.plot(df['time'], df['x'], color='black', label='reservoir head (m)', linestyle='--', alpha=0.5)
    ax2.set_ylabel('reservoir head (m)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # print the value of gout to 3 significant figures in the top right of the plot in a box
    plt.text(0.95, 0.88, 'gout = ' + str(round(gout, 3)), horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Dam operation model results from ' + start_date + ' to ' + end_date)
    plt.show()

    #save the figure as fig_name + '.png' in the plots folder
    fig.savefig('plots/' + fig_name + '.png')
    
    return inflow, x, w, r, gout



# test the updated model
#damop_model_UPDATED(params_Q1a)

# test with the next dictionary for 3 months
#damop_model_UPDATED(params_Q1_2017_2months)

# test with the two month dictionary
#damop_model_UPDATED(params_Q1_2017_2months)

# test with the one month dictionary for 2017
#damop_model_UPDATED(params_Q1_2017_1month)

# test with the four month dictionary for 2018
#damop_model_UPDATED(params_Q1_2018_4months)

#print(params_Q1_2018_4months)

# test again
#damop_model_UPDATED(params_Q1_2018_4months)

# check that the data has loaded correctly
#print(params_Q1_2018_4months)

# now run for three months in 2018
#damop_model_UPDATED(params_Q1_2018_3months)

# test again
#damop_model_UPDATED(params_Q1_2018_3months)


#print(params_Q1_2018_3months)

# test again
#damop_model_UPDATED(params_Q1_2018_3months)

# now for two months in 2018
#damop_model_UPDATED(params_Q1_2018_1month)

# test that the Q2 varaibles are correct
# print(params_Q2_small_range)
# print(params_Q2_large_range)

# # run the damop model for low tau in q2
# #amop_model_UPDATED(params_Q2_high_tau)

# # run the damop model for small range
# damop_model_UPDATED(params_Q2_small_range)
# damop_model_UPDATED(params_Q2_large_range)


# we want to run the damop model using the calibrated //
# S2S forecasting data for the 20180621 forecast
# this contains 50 ensemble members
# we will use the data from the 20180621 folder
# where the files are named as follows:
# 20180621\japan_ECMF_PF.20180621.1.diff.converted-units.study_area.mean.calibrated.nc
# 20180621\japan_ECMF_PF.20180621.2.diff.converted-units.study_area.mean.calibrated.nc
# 20180621\japan_ECMF_PF.20180621.3.diff.converted-units.study_area.mean.calibrated.nc
# etc. up to 50
# for this we want to use the original damop_model function
# which takes runoffarr, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma as inputs
# and returns inflow, x, w, r, gout as outputs

# %%

# first we need to load the data and extract the runoffarr
# we will use xarray and dask to do this
import xarray as xr
import dask.array as da

# we want to load the data from the 20180621 folder
# so we will have to loop through the files
# and extract the runoffarr from each file
# and then concatenate them together
# we will use the glob module to get the list of files
import glob

# get the list of files
# this will return a list of strings
# each string is the full path to a file
# we want to get the files from the 20180621 folder
# so we will use the os module to get the current working directory
# and then add the 20180621 folder to the end
import os

# get the current working directory
cwd = os.getcwd()
# add the 20180621 folder to the end
# this will be the path to the 20180621 folder
path = cwd + '\\20180621\\'
print(path)

# get the list of files
# this will return a list of strings
# each string is the full path to a file
# we want to get the files from the 20180621 folder
# so we will use the os module to get the current working directory
# and then add the 20180621 folder to the end
# we will use the glob module to get the list of files
# we want to get all the files that end in .nc
# so we will use the wildcard * to get all the files
# and then add .nc to the end
# this will return a list of strings
# each string is the full path to a file
list = glob.glob(path + '*.nc')
print(list)

# we want to load the data from the 20180621 folder
# so we will have to loop through the files
# 

# create an empty array to store the data for all 50 ensemble members
# the runoff data has dimensions of: (46, 1, 1, 1)
# we want to create an array with dimensions of: (50, 46, 1, 1, 1)
# which stores the ensemble member number and time
# we will use the dask module to create the array

# create an empty array to store the data for all 50 ensemble members
runoffarr_all_members = da.zeros((50, 46, 1, 1, 1))

# we will loop through the list of files
# and load each file
for i in range(0, len(list)):
    # get the file path
    file = list[i]
    #print(file)

    # load the data
    ds = xr.open_dataset(file, chunks={'time': 1})

    # extract the runoffarr
    runoffarr = ds['ro'].data

    # print the shape of this
    #print(runoffarr.shape)
    #print(runoffarr)

    # for each ensemble member we want to add the runoffarr values to an array
    runoffarr_all_members[i, :, :, :, :] = runoffarr

# print the shape of this
print(runoffarr_all_members.shape)

# make sure that the values for each ensemble member are the different
# we will use the da.mean function to calculate the mean
# print(runoffarr_all_members[1, :, :, :, :].mean().compute())
# print(runoffarr_all_members[2, :, :, :, :].mean().compute())
# print(runoffarr_all_members[3, :, :, :, :].mean().compute())
# print(runoffarr_all_members[4, :, :, :, :].mean().compute())
# print(runoffarr_all_members[5, :, :, :, :].mean().compute())

# now we want to initialize the variables needed for the damop model

# which takes runoffarr, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma as inputs
# and returns inflow, x, w, r, gout as outputs

# initialize the constants
# these are the same for all ensemble members
dt = params_Q1a['dt']
catcharea = params_Q1a['catchment_area']
kappa = params_Q1a['kappa']
hmax = params_Q1a['H_max']
hmin = params_Q1a['H_min']
wmax = params_Q1a['W_max']
wmin = params_Q1a['W_min']
rmax = params_Q1a['R_max']
sigma = params_Q1a['sigma']

# initiazize dask arrays to store the output
# for each ensemble member we want to store the inflow, x, w, r, gout
# inflow is returned as a 2D array
# with dimensions of (46, 1)
# same for x, w and r
# gout is just a single value for each ensemble member

# lets run the damop model for the first ensemble member
# just to test the output

# define runoffarr for the first ens member
runoffarr = runoffarr_all_members[0, :, :, :, :].flatten()

# # print the mean value for this
# print(runoffarr.mean().compute())

# initialize empty dask arrays to store output
# inflow = da.zeros(46,1)
# x = da.zeros(46,1)
# w = da.zeros(46,1)
# r = da.zeros(46,1)
# gout = da.zeros(1)

# call the damop function
#inflow, x, w, r, gout = damop_model(runoffarr, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)

# # look at the shape of the output
# print(inflow.shape)
# print(x.shape)
# print(w.shape)
# print(r.shape)
# print(gout.shape)

# for a single ensemble member we get output in the shapes
# inflow = (46,)
# x = (46,)
# w = (46,)
# r = (46,)
# gout = ()

# we want to store the output for each ensemble member
# so we will need to create an array to store the output
# for each ensemble member
# for inflow, x, w, r we will create an array with dimensions of (50, 46, 1)
# for gout we will create an array with dimensions of (50, 1)

# initialize dask arrays to store the output
# inflow = da.zeros((50, 46, 1))
# x = da.zeros((50, 46, 1))
# w = da.zeros((50, 46, 1))
# r = da.zeros((50, 46, 1))

# # and for gout
# gout = da.zeros(50, 1)

# create an empty array to store the data for all 50 ensemble members
# the runoff data has dimensions of: (46, 1, 1, 1)
# we want to create an array with dimensions of: (50, 46, 1, 1, 1)
# which stores the ensemble member number and time
# we will use the dask module to create the array
# inflow_all_members = da.zeros((3, 46, 1))
# x_all_members = da.zeros((3, 46, 1))
# w_all_members = da.zeros((3, 46, 1))
# r_all_members = da.zeros((3, 46, 1))
# gout_all_members = da.zeros(3)

# # now we want to loop through the ensemble members
# # when calling the damop_model function
# # we want to pass in the runoffarr for each ensemble member
# # and then store the output for each ensemble member
# for i in range(0, 2):
#     # get the runoffarr for this ensemble member
#     runoffarr = runoffarr_all_members[i, :, :, :, :].flatten()

#     # initialize empty dask arrays to store output
#     inflow = da.zeros(46,1)
#     x = da.zeros(46,1)
#     w = da.zeros(46,1)
#     r = da.zeros(46,1)
#     gout = da.zeros(1)

#     # call the damop function
#     inflow, x, w, r, gout = damop_model(runoffarr, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)

#     # store the output for this ensemble member
#     inflow_all_members[i,:,:] = inflow.reshape(46,1)
#     x_all_members[i,:,:] = x.reshape(46,1)
#     w_all_members[i,:,:] = w.reshape(46,1)
#     r_all_members[i,:,:] = r.reshape(46,1)
#     gout_all_members[i] = gout

# print the values for the first ensemble member
# print(inflow_all_members[0,:,:].compute())
# print(x_all_members[0,:,:].compute())
# print(w_all_members[0,:,:].compute())
# print(r_all_members[0,:,:].compute())
# print(gout_all_members[0].compute())

# # print the values for the second ensemble member
# print(inflow_all_members[1,:,:].compute())
# print(x_all_members[1,:,:].compute())
# print(w_all_members[1,:,:].compute())
# print(r_all_members[1,:,:].compute())
# print(gout_all_members[1].compute())


# now we've tested the damop model for a two ensemble members
# we want to run the damop model for all 50 ensemble members
# and store the output for each ensemble member
# however, we want to do this in parallel
# so we will use the dask.distributed module
# to run the damop model for each ensemble member in parallel

# # initialize the array to store the output for each ensemble member first
# inflow_all_members = da.zeros((50, 46, 1))
# x_all_members = da.zeros((50, 46, 1))
# w_all_members = da.zeros((50, 46, 1))
# r_all_members = da.zeros((50, 46, 1))
# gout_all_members = da.zeros(50)

# # now we want to run the damop model for each ensemble member in parallel
# # we will use the dask.distributed module
# # to run the damop model for each ensemble member in parallel
# # we will use the Client() function to create a client
# # this will create a local cluster
# # and connect to it
# # we will then use the client to run the damop model for each ensemble member
# # in parallel
# # we will use the client.map() function to run the damop model for each ensemble member
# # we will pass in the damop_model function
# # and the runoffarr for each ensemble member
# # we will also pass in the dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma
# # as these are constant for all ensemble members
# # we will then store the output for each ensemble member
# # in the inflow_all_members, x_all_members, w_all_members, r_all_members, gout_all_members arrays
# # we will use the client.gather() function to gather the output for each ensemble member
# # and store it in the inflow_all_members, x_all_members, w_all_members, r_all_members, gout_all_members arrays
# # we will then use the client.close() function to close the client
# # and disconnect from the cluster
# # we will then use the client.restart() function to restart the client
# # and reconnect to the cluster
# # we will then use the client.close() function to close the client
# # and disconnect from the cluster

# # import the dask.distributed module
# from dask.distributed import Client

# # create a client
# client = Client()

# # run the damop model for each ensemble member in parallel
# # and store the output for each ensemble member
# # in the inflow_all_members, x_all_members, w_all_members, r_all_members, gout_all_members arrays
# inflow_all_members = client.map(damop_model, runoffarr_all_members, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)
# x_all_members = client.map(damop_model, runoffarr_all_members, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)
# w_all_members = client.map(damop_model, runoffarr_all_members, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)
# r_all_members = client.map(damop_model, runoffarr_all_members, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)
# gout_all_members = client.map(damop_model, runoffarr_all_members, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)

# import dask.array as da
# from dask.distributed import Client
# from dask import delayed

# client = Client() # create a Dask client

# # we will use the dask module to create the array
# inflow_all_members = da.zeros((50, 46, 1))
# x_all_members = da.zeros((50, 46, 1))
# w_all_members = da.zeros((50, 46, 1))
# r_all_members = da.zeros((50, 46, 1))
# gout_all_members = da.zeros(50)

# # create an empty list to store the delayed objects
# delayed_objects = []

# # now we want to loop through the ensemble members
# # when calling the damop_model function
# # we want to pass in the runoffarr for each ensemble member
# # and then store the output for each ensemble member
# for i in range(0, 50):
#     # get the runoffarr for this ensemble member
#     runoffarr = runoffarr_all_members[i, :, :, :, :].flatten()


#     # initialize empty dask arrays to store output
#     inflow = da.zeros(46,1)
#     x = da.zeros(46,1)
#     w = da.zeros(46,1)
#     r = da.zeros(46,1)
#     gout = da.zeros(1)

#     # wrap the damop function with the delayed decorator
#     # this returns a lazy Dask object that can be parallelized
#     inflow, x, w, r, gout = delayed(damop_model)(runoffarr, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)

#     # store the output for this ensemble member
#     inflow_all_members[i,:,:] = inflow.reshape(46,1)
#     x_all_members[i,:,:] = x.reshape(46,1)
#     w_all_members[i,:,:] = w.reshape(46,1)
#     r_all_members[i,:,:] = r.reshape(46,1)
#     gout_all_members[i] = gout

#     # append the Dask objects to the list
#     delayed_objects.append(inflow)
#     delayed_objects.append(x)
#     delayed_objects.append(w)
#     delayed_objects.append(r)
#     delayed_objects.append(gout)

# # use the compute method to execute the Dask objects in parallel
# # and return the results as a list
# results = da.compute(*delayed_objects)


# now for the more simple version which should actually run
# for 50 ensemble members

# # we will use the dask module to create the array
# inflow_all_members = da.zeros((50, 46, 1))
# x_all_members = da.zeros((50, 46, 1))
# w_all_members = da.zeros((50, 46, 1))
# r_all_members = da.zeros((50, 46, 1))
# gout_all_members = da.zeros(50)


# # import the time module
# import time

# # now we want to loop through the ensemble members
# # when calling the damop_model function
# # we want to pass in the runoffarr for each ensemble member
# # and then store the output for each ensemble member
# for i in range(0, 50):
#     # get the runoffarr for this ensemble member
#     runoffarr = runoffarr_all_members[i, :, :, :, :].flatten()

#     # initialize empty dask arrays to store output
#     inflow = da.zeros(46,1)
#     x = da.zeros(46,1)
#     w = da.zeros(46,1)
#     r = da.zeros(46,1)
#     gout = da.zeros(1)

#     # call the damop function
#     # time how long it takes to run
#     start = time.time()

#     print('starting damop model for ensemble member ' + str(i))

#     inflow, x, w, r, gout = damop_model(runoffarr, dt, catcharea, kappa, hmax, hmin, wmax, wmin, rmax, sigma)
#     end = time.time()

#     print('damop model for ensemble member ' + str(i) + ' took ' + str(end - start) + ' seconds')

#     print('estimated time remaining: ' + str((end - start) * (50 - i)) + ' seconds')

#     # store the output for this ensemble member
#     inflow_all_members[i,:,:] = inflow.reshape(46,1)
#     x_all_members[i,:,:] = x.reshape(46,1)
#     w_all_members[i,:,:] = w.reshape(46,1)
#     r_all_members[i,:,:] = r.reshape(46,1)
#     gout_all_members[i] = gout

# now this had run we want to save the output to a file
# for inflow_all_members, x_all_members, w_all_members, r_all_members, gout_all_members

# set up a plot to show the results
# for inflow_all_members, x_all_members, w_all_members, r_all_members, gout_all_members

# plot the results for inflo_all_members
# set up the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# plot the results using a for loop to loop through the ensemble members
for i in range(0, 50):
    ax.plot(inflow_all_members[i, :, 0])

# add a legend



# %%
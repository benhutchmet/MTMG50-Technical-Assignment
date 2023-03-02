# contains the dictionaries used in the program

# we will probably want to add a task flag
# and then names for the plots to be saved

# define the dictionary for Q1a
params_Q1a = {'H_dam': 161, # dam height in m
                'reservoir_area': 1.3e7, # reservoir area in m^2
                'kappa': (1.3e7)/2, # proportionality constant between head and reservoir volume
                'catchment_area': 2.54e8, # catchment area in m^2
                'H_max': 0.5*161, # maximum safe dam height in m
                'H_min': 0.2*161, # minimum allowed head of water in m
                'tau': 180*86400, # reservoir emptying timescale at max flow rate (s)
                'W_max': ((1.3e7)/2)/(180*86400) * 161, # maximum flow rate in m^3/s
                'W_min': 0.1*(((1.3e7)/2)/(180*86400) * 161), # minimum flow rate in m^3/s
                'R_max': 0.2*(((1.3e7)/2)/(180*86400) * 161), # maximum relief flow avoiding turbines in m^3/s
                'G_max': 153, # maximum power generation in MW
                'mu': 153/(0.9*(((1.3e7)/2)/(180*86400) * 161)*161), # maximum power generation rate by turbines
                'sigma': 0.9, # efficiency of power generation
                'start_date': '2017-06-01', # start date for period
                'end_date': '2017-09-30', # end date for period
                'dt': 86400, # time step in seconds for conversion of runoff data
                'path': r'C:\Users\Ben Hutchins\OneDrive - University of Reading\Documents\GitHub\MTMG50-Technical-Assignment\japan_ERA5land_20170601-20190801_tokuyama.nc', # path to the netCDF file
                }

params_Q1_2017_3months = {'H_dam': 161, # dam height in m
                'reservoir_area': 1.3e7, # reservoir area in m^2
                'kappa': (1.3e7)/2, # proportionality constant between head and reservoir volume
                'catchment_area': 2.54e8, # catchment area in m^2
                'H_max': 0.5*161, # maximum safe dam height in m
                'H_min': 0.2*161, # minimum allowed head of water in m
                'tau': 180*86400, # reservoir emptying timescale at max flow rate (s)
                'W_max': ((1.3e7)/2)/(180*86400) * 161, # maximum flow rate in m^3/s
                'W_min': 0.1*(((1.3e7)/2)/(180*86400) * 161), # minimum flow rate in m^3/s
                'R_max': 0.2*(((1.3e7)/2)/(180*86400) * 161), # maximum relief flow avoiding turbines in m^3/s
                'G_max': 153, # maximum power generation in MW
                'mu': 153/(0.9*(((1.3e7)/2)/(180*86400) * 161)*161), # maximum power generation rate by turbines
                'sigma': 0.9, # efficiency of power generation
                'start_date': '2017-07-01', # start date for 3-month period
                'end_date': '2017-09-30', # end date for 3-month period
                'dt': 86400, # time step in seconds for conversion of runoff data
                'path': r'C:\Users\Ben Hutchins\OneDrive - University of Reading\Documents\GitHub\MTMG50-Technical-Assignment\japan_ERA5land_20170601-20190801_tokuyama.nc', # path to the netCDF file
                }

# test for merge dictionary operator
# extreme rainfall event occurs between 28 June - 8th July

params_Q1_2017_3months_test = params_Q1a | {'start_date': '2017-07-01'} # update only the start date value
# july august september 2017 

params_Q1_2017_2months_test = params_Q1a | {'start_date': '2017-07-01', 'end_date': '2017-08-30'} # july august 2017

params_Q1_2017_1month_test = params_Q1a | {'start_date': '2017-07-01', 'end_date': '2017-07-30'} # july 2017

params_Q1_2018_4months_test = params_Q1a | {'start_date': '2018-06-01', 'end_date': '2018-09-30'}
# june to end of sept 2018

params_Q1_2018_3months_test = params_Q1a | {'start_date': '2018-07-01'}
# july august sept 2018

params_Q1_2018_2months_test = params_Q1a | {'start_date': '2018-07-01', 'end_date': '2018-08-30'} # july august 2018

params_Q1_2018_1month_test = params_Q1a | {'start_date': '2018-07-01', 'end_date': '2018-07-30'} # july 2018





# %%
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
                'path': r'C:\Users\Ben Hutchins\OneDrive - University of Reading\Documents\GitHub\MTMG50-Technical-Assignment\tokuyama_daymean.nc', # path to the netCDF file
                'task': 'Q1', # task flag
                'fig_name': '2017-06-01-2017-09-30_optimization.png' # name of the figure to be saved
                }


# test for merge dictionary operator
# extreme rainfall event occurs between 28 June - 8th July

params_Q1_2017_3months = params_Q1a | {'start_date': '2017-07-01', 'fig_name': '2017-07-01-2017-09-30_optimization.png'} # update only the start date value
# july august september 2017 

params_Q1_2017_2months = params_Q1a | {'start_date': '2017-07-01', 'end_date': '2017-08-30', 'fig_name': '2017-07-01-2017-08-30_optimization.png'} # july august 2017

params_Q1_2017_1month = params_Q1a | {'start_date': '2017-07-01', 'end_date': '2017-07-30', 'fig_name': '2017-07-01-2017-07-30_optimization.png'} # july 2017

params_Q1_2018_4months = params_Q1a | {'start_date': '2018-06-01', 'end_date': '2018-09-30', 'fig_name': '2018-06-01-2018-09-30_optimization.png'}
# june to end of sept 2018

params_Q1_2018_3months = params_Q1_2018_4months | {'start_date': '2018-07-01', 'fig_name': '2018-07-01-2018-09-30_optimization.png'}
# july august sept 2018

params_Q1_2018_2months = params_Q1_2018_4months | {'start_date': '2018-07-01', 'end_date': '2018-08-30', 'fig_name': '2018-07-01-2018-08-30_optimization.png'} # july august 2018

params_Q1_2018_1month = params_Q1_2018_4months | {'start_date': '2018-07-01', 'end_date': '2018-07-30', 'fig_name': '2018-07-01-2018-07-30_optimization.png'} # july 2018


# now we define the dictionary for Q2
# in this question we vary H_max, H_min and W_max, which are dependent on tau, the timescale

params_Q2_tau = {'H_dam': 161, # dam height in m
                'reservoir_area': 1.3e7, # reservoir area in m^2
                'kappa': (1.3e7)/2, # proportionality constant between head and reservoir volume
                'catchment_area': 2.54e8, # catchment area in m^2
                'H_max': 0.5*161, # maximum safe dam height in m
                'H_min': 0.2*161, # minimum allowed head of water in m
                'low_tau': 90*86400, # reservoir emptying timescale at max flow rate (s)
                'high_tau': 360*86400, # reservoir emptying timescale at max flow rate (s)
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
                'path': r'C:\Users\Ben Hutchins\OneDrive - University of Reading\Documents\GitHub\MTMG50-Technical-Assignment\tokuyama_daymean.nc', # path to the netCDF file
                'task': 'Q2', # task flag
                'fig_name': '2017-06-01-2017-09-30_optimization_Q2.png' # name of the figure to be saved
                }

# define the tau's
low_tau = params_Q2_tau['low_tau']
high_tau = params_Q2_tau['high_tau']

params_Q2_low_tau = params_Q2_tau | {'tau': low_tau, 'W_max': ((1.3e7)/2)/(low_tau) * 161, 'fig_name': '2017-06-01-2017-09-30_optimization_Q2_Wmax_low_tau.png'}

# now for a high tau
params_Q2_high_tau = params_Q2_tau | {'tau': high_tau, 'W_max': ((1.3e7)/2)/(high_tau) * 161, 'fig_name': '2017-06-01-2017-09-30_optimization_Q2_Wmax_high_tau.png'}

# now modify params_Q2_tau to have a high H_max and a low h_min

params_Q2_large_range = params_Q2_tau | {'H_max': 0.8*161, 'H_min': 0.1*161, 'fig_name': '2017-06-01-2017-09-30_optimization_Q2_large_range.png'}

# now modify params_Q2_tau to have a low H_max and a high h_min

params_Q2_small_range = params_Q2_tau | {'H_max': 0.3*161, 'H_min': 0.4*161, 'fig_name': '2017-06-01-2017-09-30_optimization_Q2_small_range.png'}

# %%

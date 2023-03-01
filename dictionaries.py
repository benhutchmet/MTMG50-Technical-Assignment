# %%
# contains the dictionaries used in the program

# define the dictionary for Q1
params_taskA = {'H_dam': 161, # dam height in m
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
                'path': r'C:\Users\Ben Hutchins\OneDrive - University of Reading\Documents\GitHub\MTMG50-Technical-Assignment\japan_ERA5land_20170601-20190801_tokuyama.nc', # path to the netCDF file
                }

# %%

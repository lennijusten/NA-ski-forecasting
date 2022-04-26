import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import colors
import xarray as xr
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cpf
import cartopy.feature as cfeature
import pickle
import glob
from natsort import natsorted
import numpy as np
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xesmf
from matplotlib.gridspec import GridSpec
import scipy.signal as sig
import os
from tqdm import tqdm
from dask.distributed import Client, LocalCluster
import dask
import earthpy as et
import regionmask
import geopandas as gpd
# import geoplot.crs as gcrs
import pandas as pd
import cmaps
import seaborn as sns
import joypy
# from mpl_toolkits.axes_grid1.inset≤_locator import inset_axes


wetbulb_path = '/adhara_a/ljusten/Wetbulb-bc/*.nc'
temp_path = '/adhara_a/ljusten/TREFHT-bc-degC/*.nc'

ensmems = list(range(1, 36, 1)) + list(range(101, 106, 1))

# Consider looping through each member 
# Save daily climatology for each member as a datarray 

def init_data(temp_path, wetbulb_path, unit='C',n_mems=None):
    if n_mems is None:
        n_mems = len(ensmems)
    else:
        pass
    
    openme_temp = natsorted(glob.glob(temp_path))
    openme_wetbulb = natsorted(glob.glob(wetbulb_path))
    
    temp = []
    wetbulb = []
    for i,mem in tqdm(enumerate(ensmems[0:n_mems])):
        temp_iter = xr.open_dataset(openme_temp[i],chunks={})
        wetbulb_iter = xr.open_dataset(openme_wetbulb[i],chunks={})
        
        if unit is 'C':
            pass
        elif unit is 'K':
            print('This version is slow and not yet fully optimized')
            temp_iter['TREFHT'] = temp_iter['TREFHT'] + 273.15
            wetbulb_iter['wetbulb'] = wetbulb_iter['wetbulb'] + 273.15
        else:
            print("You will find no support (i.e. love) for Farenheit here. Trying using 'C' for Celsius or 'K' for Kelvin")
        
        temp.append(temp_iter)
        wetbulb.append(wetbulb_iter)
        
    temp_agg = xr.concat(temp, dim='M', compat='override', coords='minimal', data_vars=['TREFHT'])
    wetbulb_agg = xr.concat(wetbulb, dim='M', compat='override', coords='minimal', data_vars=['wetbulb'])
        
    return temp_agg, wetbulb_agg

temp, wetbulb = init_data(temp_path, wetbulb_path,unit='C',n_mems=40)

snow_thresh = -2.0
coords = [[45,268],[39,253],[41,253],[44,289]]
start_months = [10,11,12,1] # months to look for start date

# Initialize df with year index
df = pd.DataFrame()
df.index = np.arange(1920,2101)

def select_coord(var, lat, lon):
    return var.sel(lat=lat,lon=lon)

wb = select_coord(wetbulb,coords[3][0],coords[3][1])
wetbulb_bool = wb <= snow_thresh # Day below snowthresh (T/F)

wb_start = wetbulb_bool.sel(time=wetbulb_bool.time.dt.month.isin(start_months))
datetimeindex = wb_start.indexes['time'].to_datetimeindex() # convert cftime to datetime
wb_start['time'] = pd.to_datetime(datetimeindex)

# Add custom group by year functionality
custom_year = wb_start['time'].dt.year

time1 = [pd.Timestamp(i) for i in custom_year['time'].values] # convert time type to pd.Timestamp
time2 = [i + relativedelta(years=1) if i.month>=10 else i for i in time1]
wb_start['time'] = time2

wb_start1 = wb_start.groupby('time.year')


smd_thresh = 5
mean_first_date = []
std_first_date = []

for start_year in tqdm(wb_start1):
    x1 = start_year[1]['wetbulb'].cumsum('time')
    time = start_year[1]['time']

    mem_mean = []
    for mem in range(len(start_year[1].M)):
        try:
            fd = time[x1.sel(M=mem)==smd_thresh][0].values
            mem_mean.append(pd.to_datetime(fd))
        except:
            pass
            

    mem_mean = pd.Series(mem_mean)
    mean_dt_temp = mem_mean.pipe(lambda d: (lambda m: m + (d - m).mean())(d.min())).to_pydatetime()
    mean_dt = datetime(*mean_dt_temp.timetuple()[:3]) 

    # start_mean_idx = round(np.mean(np.array([i.day for i in mem_mean])))
    start_std_idx = round(np.std(np.array([i.day for i in mem_mean])))
    
    mean_first_date.append(mean_dt)
    std_first_date.append(start_std_idx)
    
    
# Store data (serialize)
with open('mean_first_date-coord3.pkl', 'wb') as handle:
    pickle.dump(mean_first_date, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('std_first_date-coord3.pkl', 'wb') as handle:
    pickle.dump(std_first_date, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

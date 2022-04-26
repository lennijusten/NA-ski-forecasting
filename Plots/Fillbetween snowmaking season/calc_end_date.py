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
start_months = [10,11,12] # months to look for start date
end_months = [2,3,4,5] # months to look for end date

# Initialize df with year index
df = pd.DataFrame()
df.index = np.arange(1920,2101)

def select_coord(var, lat, lon):
    return var.sel(lat=lat,lon=lon)

wb = select_coord(wetbulb,coords[0][0],coords[0][1])
wetbulb_bool = wb <= snow_thresh # Day below snowthresh (T/F)

wb_end = wetbulb_bool.sel(time=wetbulb_bool.time.dt.month.isin(end_months))
datetimeindex = wb_end.indexes['time'].to_datetimeindex() # convert cftime to datetime
wb_end['time'] = pd.to_datetime(datetimeindex)

wb_end1 = wb_end.groupby('time.year')

mean_end_date = []
std_end_date = []

for end_year in tqdm(wb_end1):
    x1 = end_year[1]['wetbulb']
    time = end_year[1]['time']
    
    mem_mean = []
    for mem in range(len(end_year[1].M)):
        try:
            ld_idx = np.where(x1.sel(M=mem).values == True)[0][-1]
        except:
            ld_idx = np.where(x1.sel(M=mem).values == True)[0]
        try:    
            ld = time[ld_idx].values
        except:
            continue
        
        mem_mean.append(ld)        
 
    mem_mean = pd.Series(mem_mean)
    
    mean_dt_temp = mem_mean.pipe(lambda d: (lambda m: m + (d - m).mean())(d.min())).to_pydatetime()
    mean_dt = datetime(*mean_dt_temp.timetuple()[:3]) 

    # start_mean_idx = round(np.mean(np.array([i.day for i in mem_mean])))
    end_std_idx = round(np.std(np.array([i.day for i in mem_mean])))
    
    mean_end_date.append(mean_dt)
    std_end_date.append(end_std_idx)
    
    
# Store data (serialize)
with open('mean_last_date-coord0.pkl', 'wb') as handle:
    pickle.dump(mean_end_date, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('std_last_date-coord0.pkl', 'wb') as handle:
    pickle.dump(std_end_date, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

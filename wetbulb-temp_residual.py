# Calculate and visualize the residual between temperature and wetbulb temperature

# Author: Lennart Justen

import numpy as np
import matplotlib.pyplot as plt
import metpy.calc
from metpy.units import units
from tqdm import tqdm
import pickle
import xarray as xr

# Ensemble member 1
f_temp1 = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/TREFHT/b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h1.TREFHT.18500101-20051231.nc'
f_temp2 = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/TREFHT/b.e11.BRCP85C5CNBDRD.f09_g16.001.cam.h1.TREFHT.20060101-20801231.nc'
f_temp3 = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/TREFHT/b.e11.BRCP85C5CNBDRD.f09_g16.001.cam.h1.TREFHT.20810101-21001231.nc'

f_wetbulb = '/home1/ljusten/CESM/Wetbulb/b.e11.BRCP85C5CNBDRD.f09_g16.001.cam.h1.WETBULB.18500101-21001231.nc'


def init_data(f_temp, f_wetbulb):  # pass as lists
    wetbulb = xr.open_dataset(f_wetbulb)
    # wetbulb = wetbulb.rename_vars({"__xarray_dataarray_variable__": "wetbulb"})

    temp_ds = []
    for i in f_temp:
        temp = xr.open_dataset(i)
        temp_ds.append(temp)

    return xr.concat(temp_ds, dim="time"), wetbulb


temp, wetbulb = init_data([f_temp1, f_temp2, f_temp3], f_wetbulb)


def select_grid(temp, bot_lat, top_lat, left_lon, right_lon):  # Wetbulb is calculated over this grid by default
    ds_temp = temp.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    return ds_temp


temp1 = select_grid(temp, 23, 72, 190, 295)

temp_degC = temp1['TREFHT'].values - 273.16

# Residual grid of the same shape
residual = temp_degC-wetbulb['wetbulb'].values

# Plot time mean of residual
lon=temp1['lon']
lat=temp1['lat']
plt.pcolormesh(lon,lat,np.mean(residual, axis=0))
plt.title('mean Temp-Wetbulb residual (degC) from 1850-2100')
plt.colorbar()
plt.show()

# Mean absolute residual
MAE = np.sum(residual.flatten())/len(residual.flatten())


# Check the reference grid method against the metpy wetbulb method
# https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.wet_bulb_temperature.html

# Author: Lennart Justen

import numpy as np
import matplotlib.pyplot as plt
import metpy.calc
from metpy.units import units
from tqdm import tqdm
import pickle
import xarray as xr


def init_data():
    ds_press = xr.open_dataset(
        '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/monthly/PS/b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h0.PS.185001-200512.nc')
    ds_temp = xr.open_dataset(
        '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/TREFHT/b.e11.BRCP85C5CNBDRD.f09_g16.001.cam.h1.TREFHT.20060101-20801231.nc')
    ds_qbot = xr.open_dataset(
        '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/QBOT/b.e11.BRCP85C5CNBDRD.f09_g16.001.cam.h1.QBOT.20060101-20801231.nc')

    return ds_temp, ds_press, ds_qbot


temp, press, qbot = init_data()

def select_grid(temp, press, qbot, bot_lat, top_lat, left_lon, right_lon):
    ds_temp = temp.sel(lat=slice(bot_lat,top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_press = press.sel(lat=slice(bot_lat,top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_qbot = qbot.sel(lat=slice(bot_lat,top_lat)).sel(lon=slice(left_lon, right_lon))
    return ds_temp, ds_press, ds_qbot

temp1, press1, qbot1 = select_grid(temp, press, qbot,23,72,190,295)

press_mean = press1.mean('time')
press_mean['PS'] = press_mean.PS.assign_attrs(units='Pa')

def select_time(temp, qbot, start, end):
    ds_temp = temp.sel(time=slice(start, end))
    ds_qbot = qbot.sel(time=slice(start, end))
    return ds_temp, ds_qbot

temp2, qbot2 = select_time(temp1, qbot1, '2007-01-01 00:00:00', '2007-02-01 00:00:00')


def get_dewpoint(temp, press, qbot, method):
    if method == 'RH':
        # RH = metpy.calc.relative_humidity_from_mixing_ratio(qbot['QBOT'], temp['TREFHT'], press['PS'])
        RH = metpy.calc.relative_humidity_from_mixing_ratio(press['PS'], temp['TREFHT'], qbot['QBOT'])
        # dewpoint = metpy.calc.dewpoint_from_relative_humidity(temp['TREFHT'], RH)
        dewpoint = metpy.calc.dewpoint_from_relative_humidity(temp['TREFHT'], RH)
        return RH, dewpoint
    elif method == 'WVPP':
        vapor_pressure = metpy.calc.vapor_pressure(press['PS'], qbot[
            'QBOT'])  # https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.vapor_pressure.html
        dewpoint = metpy.calc.dewpoint(vapor_pressure)
        return vapor_pressure, dewpoint
    else:
        print("Unknown method--Relative Humidity: 'RH', Water Vapor Partial Pressure: 'WVPP'")


RH, dewpoint = get_dewpoint(temp2, press_mean, qbot2, 'RH')


wetbulb_metpy = metpy.calc.wet_bulb_temperature(press_mean['PS'], temp2['TREFHT'], dewpoint)

with open('wetbulb_metpy_2006-02-02--2016-02-02.pkl', 'rb') as f:
    pickle.dump(wetbulb_metpy, f)

temp_degC = temp2['TREFHT'].values - 273.15
wb_metpy_degC = wetbulb_metpy.values - 273.15
wb_metpy_degC = np.moveaxis(wb_metpy_degC,2,0)

with (open("wetbulb_metpy_lowres_2_15_grid.pkl", "rb")) as openfile:
    while True:
        try:
            wetbulb = pickle.load(openfile)
        except EOFError:
            break

wetbulb_key = np.array(wetbulb)  # Global lowres grid

press_min = 51000 # Pa
press_max = 105000 # Pa
press_nlayers = 200

temp_min = -81.0 # degC
temp_max = 48.0 # degC
temp_res = 1.0

dewpoint_min = -81.0
dewpoint_max = 30.0
dewpoint_res = 1.0

# Range of param values (Global)
p_space = np.linspace(press_min, press_max, press_nlayers)
t_space = np.arange(temp_min, temp_max, temp_res)
dp_space = np.arange(dewpoint_min, dewpoint_max, dewpoint_res)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

long_temp = temp_degC.flatten()
long_press = np.tile(press_mean['PS'].values.flatten(), temp_degC.shape[0])
long_dewpoint = dewpoint.values.flatten()

wetbulb_search = []
for i in range(len(long_temp)):
    p_idx = find_nearest(p_space, long_press[i])
    t_idx = find_nearest(t_space, long_temp[i])
    dp_idx = find_nearest(dp_space, long_dewpoint[i])
    wetbulb_search.append(round(wetbulb_key[p_idx][dp_idx][t_idx],3))


wetbulb_search = np.reshape(np.array(wetbulb_search),temp_degC.shape)

residual = wetbulb_search - wb_metpy_degC
print('max residual = {}'.format(np.max(residual)))

MAE = np.sum(abs(residual.flatten()))/len(residual.flatten())
print('MAE = {}'.format(MAE))

lon=temp2['lon']
lat=temp2['lat']
plt.pcolormesh(lon,lat,np.mean(residual, axis=0))
plt.title('mean wetbulb residual (reference grid method - metpy method)')
plt.xlabel('lon')
plt.ylabel('lat')
plt.colorbar()
plt.show()

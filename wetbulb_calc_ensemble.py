# Calculates wetbulb for an entire ensemble member and returns a single netCDF file
# Runtime ~1hour

import numpy as np
import matplotlib.pyplot as plt
import metpy.calc
from metpy.units import units
from tqdm import tqdm
import pickle
import xarray as xr
import argparse

# let input be list of all filenames 'f_temp1,f_temp2,f_temp3,f_qbot1,f_qbot2,f_qbot3,f_press_all, wetbulb_save_filename.nc'

# Example bash
# /home1/ljusten/miniconda3/envs/climate36/bin/python /home1/ljusten/CESM/wetbulb_final_CHTC.py
# 'b.e11.B20TRC5CNBDRD.f09_g16.009.cam.h1.TREFHT.19200101-20051231.nc,
# b.e11.BRCP85C5CNBDRD.f09_g16.009.cam.h1.TREFHT.20060101-20801231.nc,
# b.e11.BRCP85C5CNBDRD.f09_g16.009.cam.h1.TREFHT.20810101-21001231.nc,
# b.e11.B20TRC5CNBDRD.f09_g16.009.cam.h1.QBOT.19200101-20051231.nc,
# b.e11.BRCP85C5CNBDRD.f09_g16.009.cam.h1.QBOT.20060101-20801231.nc,
# b.e11.BRCP85C5CNBDRD.f09_g16.009.cam.h1.QBOT.20810101-21001231.nc,
# b.e11.B20TRC5CNBDRD.f09_g16.009.cam.h0.PS.192001-200512.nc,
# b.e11.BRCP85C5CNBDRD.f09_g16.009.cam.h1.WETBULB.192001-21001231.nc' &> output_ensemble9.txt &

# Base paths
temp_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/TREFHT/'
qbot_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/QBOT/'
press_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/monthly/PS/'


def read_args():
    parser = argparse.ArgumentParser(description="wetbulb")

    parser.add_argument("input",
                        type=str
                        )

    args = parser.parse_args()
    return args


args = read_args()


def read_input(args):  # split input args into python list object
    inputs = args.input.split(',')

    return inputs


inputs = read_input(args)

# Full paths to enemble member data
f_temp1 = temp_path + inputs[0]
f_temp2 = temp_path + inputs[1]
f_temp3 = temp_path + inputs[2]

f_qbot1 = qbot_path + inputs[3]
f_qbot2 = qbot_path + inputs[4]
f_qbot3 = qbot_path + inputs[5]

# only using the time mean of the first press file although three press files exist for each ensemble member
f_press_all = press_path + inputs[6]

save_path = '/home1/ljusten/CESM/' + inputs[7]


def init_data(f_temp, f_qbot, f_press):  # pass inputs as lists. Returns concatinated xarrays
    ds_press = xr.open_dataset(f_press)

    # append all three temp and qbot files to python list
    temp_ds = []
    qbot_ds = []
    for i in range(3):
        temp = xr.open_dataset(f_temp[i])
        qbot = xr.open_dataset(f_qbot[i])
        temp_ds.append(temp)
        qbot_ds.append(qbot)

    return xr.concat(temp_ds, dim="time"), xr.concat(qbot_ds, dim="time"), ds_press


temp, qbot, press = init_data([f_temp1, f_temp2, f_temp3], [f_qbot1, f_qbot2, f_qbot3], f_press_all)


def select_grid(temp, press, qbot, bot_lat, top_lat, left_lon, right_lon):
    ds_temp = temp.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_press = press.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_qbot = qbot.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    return ds_temp, ds_press, ds_qbot


temp1, press1, qbot1 = select_grid(temp, press, qbot, 23, 72, 190, 295)  # Standard NA grid

press_mean = press1.mean('time')
press_mean['PS'] = press_mean.PS.assign_attrs(units='Pa')


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


RH, dewpoint = get_dewpoint(temp1, press_mean, qbot1, 'RH')

temp_degC = temp1['TREFHT'].values - 273.15  # Convert temp to degC (no metpy units)

with (open("wetbulb_metpy_lowres_200_1.0_grid.pkl", "rb")) as openfile:  # Load reference grid
    while True:
        try:
            wetbulb = pickle.load(openfile)
        except EOFError:
            break

wetbulb_key = np.array(wetbulb)  # Global lowres grid

# Specify params for reference grid

# LowRes Params
press_min = 51000  # Pa
press_max = 105000  # Pa
press_nlayers = 200

temp_min = -81.0  # degC
temp_max = 48.0  # degC
temp_res = 1.0

dewpoint_min = -81.0  # degC
dewpoint_max = 30.0  # degC
dewpoint_res = 1.0

# Range of param values (Global)
p_space = np.linspace(press_min, press_max, press_nlayers)
t_space = np.arange(temp_min, temp_max, temp_res)
dp_space = np.arange(dewpoint_min, dewpoint_max, dewpoint_res)


def find_nearest(array, value):  # Returns index of element in array nearest to input value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Flatten all wetbulb inputs
long_temp = temp_degC.flatten()
long_press = np.tile(press_mean['PS'].values.flatten(), temp_degC.shape[0])
long_dewpoint = dewpoint.values.flatten()

wetbulb_search = []
for i in range(len(long_temp)):
    p_idx = find_nearest(p_space, long_press[i])
    t_idx = find_nearest(t_space, long_temp[i])
    dp_idx = find_nearest(dp_space, long_dewpoint[i])
    wetbulb_search.append(wetbulb_key[p_idx][dp_idx][t_idx])  # append element of reference grid nearest too inputs

wetbulb = np.reshape(np.array(wetbulb_search), temp_degC.shape)  # reshape flattened wetbulb into original dims

wb = xr.DataArray(wetbulb, dims=['time', 'lat', 'lon'], coords={'time': temp1.time, 'lat': temp1.lat, 'lon': temp1.lon})
wb.attrs['units'] = 'degC'
wb.attrs['long_name'] = 'Wetbulb temperature'

wb = wb.rename_vars({"__xarray_dataarray_variable__": "wetbulb"})  

wb.to_netcdf(save_path)

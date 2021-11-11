# Description: calculate wetbulb temperature from the refactored files of the preprocess.py script. Wetbulb is
# calculated using the low-res reference grid.

# Author: Lenni Justen
# Date 11/10/21

import numpy as np
import metpy.calc
from tqdm import tqdm
import pickle
import xarray as xr
from natsort import natsorted
import glob

temp_path = '/adhara_a/ljusten/TREFHT/*.nc'
qbot_path = '/adhara_a/ljusten/QBOT/*.nc'
press_path = '/adhara_a/ljusten/b.e11.B20TRC5CNBDRD.f09_g16.MEAN-001.cam.h0.PS.185001-200512.nc'

ensmems = list(range(1, 36, 1)) + list(range(101, 106, 1)) # Define list of ensemble member IDs


def init_data(temp_path, qbot_path, press_path):
    # Purpose: open temp and qbot files into a list of ensemble members and return. Open the press datset and return.

    press = xr.open_dataset(press_path)
    temp = []
    qbot = []

    for ft, fq in tqdm(zip(natsorted(glob.glob(temp_path)), natsorted(glob.glob(qbot_path)))):
        temp.append(xr.open_dataset(ft))
        qbot.append(xr.open_dataset(fq))

    return temp, press, qbot


temp, press, qbot = init_data(temp_path, qbot_path, press_path)


def load_reference_grid(path):
    with (open(path, "rb")) as openfile:
        while True:
            try:
                wetbulb_grid = pickle.load(openfile)
            except EOFError:
                break

    # Params for LowRes grid
    grid_config = {
        'press_min': 51000,  # Pa
        'press_max': 105000,  # Pa
        'press_nlayers': 200,

        'temp_min': -81.0,  # degC
        'temp_max': 48.0,  # degC
        'temp_res': 1.0,

        'dewpoint_min': -81.0,
        'dewpoint_max': 30.0,
        'dewpoint_res': 1.0,
    }

    return np.array(wetbulb_grid), grid_config


wetbulb_grid, grid_config = load_reference_grid('/home1/ljusten/wetbulb_metpy_lowres_2_15_grid.pkl')


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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def run_harness(overwrite=False):
    # Range of param values (Global)
    p_space = np.linspace(grid_config['press_min'], grid_config['press_max'], grid_config['press_nlayers'])
    t_space = np.arange(grid_config['temp_min'], grid_config['temp_max'], grid_config['temp_res'])
    dp_space = np.arange(grid_config['dewpoint_min'], grid_config['dewpoint_max'], grid_config['dewpoint_res'])

    for i in tqdm(range(len(ensmems))):
        save_name = "/adhara_a/ljusten/Wetbulb/b.e11.BRCP85C5CNBDRD.f09_g16.{}.cam.h1.WETBULB.19200101-21001231.nc".format(
            ensmems[i])

        if overwrite:
            try:
                os.remove(save_name)
            except OSError:
                pass
        else:
            if os.path.exists(save_name):
                continue
            else:
                pass

        temp_iter = temp[i]
        qbot_iter = qbot[i]

        RH, dewpoint = get_dewpoint(temp_iter, press, qbot_iter, 'RH')

        temp_degC = temp_iter['TREFHT'].values - 273.15

        long_temp = temp_degC.flatten()
        long_press = np.tile(press['PS'].values.flatten(), temp_degC.shape[0])
        long_dewpoint = dewpoint.values.flatten()

        wetbulb_search = []
        for ii in range(len(long_temp)):
            p_idx = find_nearest(p_space, long_press[ii])
            t_idx = find_nearest(t_space, long_temp[ii])
            dp_idx = find_nearest(dp_space, long_dewpoint[ii])
            wetbulb_search.append(round(wetbulb_grid[p_idx][dp_idx][t_idx], 4))

        wetbulb = np.reshape(np.array(wetbulb_search), temp_degC.shape)

        wetbulb_ = xr.DataArray(wetbulb, dims=['time', 'lat', 'lon'],
                                coords={'time': temp_iter.time, 'lat': temp_iter.lat, 'lon': temp_iter.lon})
        wetbulb_.attrs['units'] = 'degC'
        wetbulb_.attrs['long_name'] = 'Wetbulb temperature'

        try:
            wetbulb_ = wetbulb_.rename({'__xarray_dataarray_variable__': "wetbulb"})
        except:
            pass

        wetbulb_.to_netcdf(save_name)


run_harness()
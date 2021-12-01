# Description: Bias correct the CESM temperature (TREFHT) and wetbulb files with gridded data from the ERA5
# observation-based model
# Author: Lennart Justen and Elizabeth Maroon

import xarray as xr
import glob
from natsort import natsorted
import numpy as np
import xesmf
import scipy.signal as sig
import os
from tqdm import tqdm
import cftime
import pickle

wetbulb_path = '/adhara_a/ljusten/Wetbulb/*.nc'
temp_path = '/adhara_a/ljusten/TREFHT/*.nc'
era5_temp_path = '/adhara_b/ERA5/daily/T2m/'
era5_dewpoint_path = '/adhara_b/ERA5/daily/Td2m/'
era5_pressure_path = '/adhara_b/ERA5/daily/surface_pressure/'
era5_wetbulb_path = '/adhara_a/ljusten/Wetbulb-era5/wetbulb_era5.nc'

ensmems = list(range(1, 36, 1)) + list(range(101, 106, 1))
openme_temp = natsorted(glob.glob(temp_path))
openme_wetbulb = natsorted(glob.glob(wetbulb_path))

# Start by calculating wetbulb temperature for the ERA5 dataset

def preprocess_era5(ds):
    try:
        dummy = ds['TREFHT']
        temp = True
    except:
        temp = False

    if temp:
        ds = ds.sel(latitude=slice(72, 23), longitude=slice(190, 295))
    else:
        pass

    ds = ds.resample(time='1D').mean('time')  # 6-hourly --> daily means
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))  # removes leap days
    doy = ds.time.dt.dayofyear
    if doy.max() == 366: doy = xr.where(doy > 60, doy - 1, doy)
    ds['doy_noleap'] = doy  # doy without leap days
    ds = ds.sortby('latitude')
    return ds


def init_era5_data(temp_path, dewpoint_path, pressure_path):
    temp = xr.open_mfdataset(sorted(glob.glob(temp_path + 'era5_global_*.nc')),
                             combine='by_coords', coords='minimal', compat='override',
                             preprocess=preprocess_era5)

    dewpoint = xr.open_mfdataset(sorted(glob.glob(dewpoint_path + 'era5_global_*.nc')),
                                 combine='by_coords', coords='minimal', compat='override',
                                 preprocess=preprocess_era5)

    pressure = xr.open_mfdataset(sorted(glob.glob(pressure_path + 'era5_global_*.nc')),
                                 combine='by_coords', coords='minimal', compat='override',
                                 preprocess=preprocess_era5)

    return temp, dewpoint, pressure


temp, dewpoint, pressure = init_era5_data(era5_temp_path, era5_dewpoint_path, era5_pressure_path)


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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def calc_wetbulb(temp, dewpoint, pressure, save_name):
    temp = temp['t2m'].values - 273.15
    dewpoint = dewpoint['d2m'].values - 273.15
    pressure = pressure['sp'].values

    long_temp = temp.flatten()
    long_dewpoint = dewpoint.flatten()
    long_pressure = pressure.flatten()

    # Range of param values (Global)
    p_space = np.linspace(grid_config['press_min'], grid_config['press_max'], grid_config['press_nlayers'])
    t_space = np.arange(grid_config['temp_min'], grid_config['temp_max'], grid_config['temp_res'])
    dp_space = np.arange(grid_config['dewpoint_min'], grid_config['dewpoint_max'], grid_config['dewpoint_res'])

    wetbulb_search = []
    for i in range(len(long_temp)):
        p_idx = find_nearest(p_space, long_pressure[i])
        t_idx = find_nearest(t_space, long_temp[i])
        dp_idx = find_nearest(dp_space, long_dewpoint[i])
        wetbulb_search.append(round(wetbulb_grid[p_idx][dp_idx][t_idx], 4))

    wetbulb = np.reshape(np.array(wetbulb_search), temp.shape)

    wetbulb_ = xr.DataArray(wetbulb, dims=['time', 'latitude', 'longitude'],
                            coords={'time': temp.time, 'lat': temp.latitude, 'lon': temp.longitude})
    wetbulb_.attrs['units'] = 'degC'
    wetbulb_.attrs['long_name'] = 'Wetbulb temperature'

    try:
        wetbulb_ = wetbulb_.rename({'__xarray_dataarray_variable__': "wetbulb"})
    except:
        pass

    wetbulb_.to_netcdf(save_name)
    return wetbulb_


wetbulb = calc_wetbulb(temp, dewpoint, pressure, era5_wetbulb_path)


# Apply bias correction to temp and wetbulb

def init_era5_dataset(era5_path, temp):
    if temp:
        ds_era5 = xr.open_mfdataset(sorted(glob.glob(era5_path + 'era5_global_*.nc')),
                                    combine='by_coords', coords='minimal', compat='override',
                                    preprocess=preprocess_era5)
        era5_climo = dayofyearmean(ds_era5)['t2m']
    else:
        ds_era5 = xr.open_dataset(era5_path)
        ds_era5 = preprocess_era5(ds_era5)
        try:
            ds_era5 = ds_era5.rename({'__xarray_dataarray_variable__': "wetbulb"})
        except:
            pass

        era5_climo = dayofyearmean(ds_era5)['wetbulb']

    era5_climo.load()

    lpfilt_era5_climo = xr.apply_ufunc(filter_wrapper, era5_climo, input_core_dims=[['doy_noleap']],
                                       output_core_dims=[['doy_noleap']],
                                       vectorize=True)

    # make sure ERA5 dims have same names as CESMLE dims
    lpfilt_era5_climo = lpfilt_era5_climo.rename({'latitude': 'lat', 'longitude': 'lon',
                                                  'doy_noleap': 'dayofyear'})

    return ds_era5, era5_climo, lpfilt_era5_climo


# calculates daily mean by day-of-year
def dayofyearmean(ds):
    doy = ds.doy_noleap
    return ds.groupby(doy).mean()


# filtering daily climo on example; code from AOS 575 application lab 5

### Function to make window for Lanczos Filter
def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.
    Args:
    window: int
        The length of the filter window in time steps
    cutoff: float
        The cutoff frequency in inverse time steps.
        (e.g., for data collected every hour, frequency units are per hour)
    """
    order = ((window - 1) // 2) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n - 1:0:-1] = firstfactor * sigma
    w[n + 1:-1] = firstfactor * sigma
    return w[1:-1]


# specify the window length for filters
window = 30 ## (default 25)
cutoff = 1. / 30. ## (default 1./11.)
wgts24 = low_pass_weights(window, cutoff)

def filter_wrapper(x):
    return sig.filtfilt(wgts24, np.sum(wgts24), x)


def run_harness(cesm_path, openme, era5_path, save_path, save_path_climo, overwrite=False, unit='K', nmems=None):
    if nmems is None:
        nmems = len(ensmems)
    else:
        pass

    try:
        dummy = openme[0]['TREFHT']
        temp = True
        substr = 'TREFHT'
        key = 'TREFHT'
    except:
        temp = False
        substr = 'WETBULB'
        key = 'wetbulb'

    inserttxt = "-bc"
    inserttxt_climo = "-climo"
    filenames_ = [os.path.basename(i) for i in natsorted(glob.glob(cesm_path))]
    filenames = [i[:i.index(substr) + len(substr)] + inserttxt + i[i.index(substr) + len(substr):] for i in filenames_]
    climo_filenames = [i[:i.index(substr) + len(substr)] + inserttxt_climo + i[i.index(substr) + len(substr):] for i in
                       filenames_]

    ds_era5, era5_climo, lpfilt_era5_climo = init_era5_dataset(era5_path, temp)

    lat_era5 = ds_era5.lat
    lon_era5 = ds_era5.lon

    ds_out = xr.Dataset({'lat': (['lat'], lat_era5.data),
                         'lon': (['lon'], lon_era5.data), })

    print(ds_out)

    for i, mem in tqdm(enumerate(ensmems[0:nmems])):
        var_iter = xr.open_dataset(openme[i])
        save_name = os.path.join(save_path, filenames[i])
        save_name_climo = os.path.join(save_path_climo, climo_filenames[i])

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

        if not temp:
            try:
                var_iter = var_iter.rename({'__xarray_dataarray_variable__': "wetbulb"})
            except:
                pass

        if unit is 'C':
            if temp:
                var_iter['TREFHT'].values = var_iter['TREFHT'].values - 273.15
            else:
                pass
        elif unit is 'K':

            var_iter['wetbulb'].values = var_iter['wetbulb'].values + 273.15
        else:
            print(
                "You will find no support (i.e. love) for Farenheit here. Trying using 'C' for Celsius or 'K' for Kelvin")

        d1 = cftime.DatetimeNoLeap(1979, 1, 1, 0, 0, 0)
        d2 = cftime.DatetimeNoLeap(2019, 12, 31, 11, 59, 59)
        cesm_climo = var_iter[key].sel(time=slice(d1, d2))
        cesm_climo = cesm_climo.groupby(cesm_climo.time.dt.dayofyear).mean()
        cesm_climo.to_netcdf(save_name_climo)
        cesm_climo.load()

        lpfilt_cesm_climo = xr.apply_ufunc(filter_wrapper, cesm_climo, input_core_dims=[['dayofyear']],
                                           output_core_dims=[['dayofyear']],
                                           vectorize=True)

        # regrid climo
        regridder = xesmf.Regridder(var_iter, ds_out, 'bilinear')

        lpfilt_cesm_climo_re = regridder(lpfilt_cesm_climo.transpose('dayofyear', 'lat', 'lon'))

        # regrid CESM original to ERA5
        cesm_re = regridder(var_iter)

        # calculate bias between one ensemble member and ERA5
        cesm_bias = lpfilt_cesm_climo_re - lpfilt_era5_climo

        cesm_bias_removed = cesm_re.groupby(cesm_re.time.dt.dayofyear) - cesm_bias

        cesm_bias_removed = cesm_bias_removed.astype('float32')
        cesm_bias_removed.to_netcdf(save_name)


era5_path = '/adhara_a/ljusten/Wetbulb-era5/wetbulb_era5.nc'
save_path = '/adhara_a/ljusten/Wetbulb-bc/'
save_path_climo = '/adhara_a/ljusten/Wetbulb-climatology/'

run_harness(wetbulb_path, openme_wetbulb, era5_path, save_path, save_path_climo, overwrite=True)

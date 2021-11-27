# Description: Bias correct the CESM temperature (TREFHT) files with gridded data from the ERA5 observation-based model
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

wetbulb_path = '/adhara_a/ljusten/Wetbulb/*.nc'
temp_path = '/adhara_a/ljusten/TREFHT/*.nc'

ensmems = list(range(1, 36, 1)) + list(range(101, 106, 1))
openme_temp = natsorted(glob.glob(temp_path))
openme_wetbulb = natsorted(glob.glob(wetbulb_path))


def init_era5_dataset(era5_path):
    ds_era5 = xr.open_mfdataset(sorted(glob.glob(era5_path + 'era5_global_*.nc')),
                                combine='by_coords', coords='minimal', compat='override',
                                preprocess=preprocess_era5)

    era5_climo = dayofyearmean(ds_era5)['t2m']
    era5_climo.load()

    lpfilt_era5_climo = xr.apply_ufunc(filter_wrapper, era5_climo, input_core_dims=[['doy_noleap']],
                                       output_core_dims=[['doy_noleap']],
                                       vectorize=True)

    # make sure ERA5 dims have same names as CESMLE dims
    lpfilt_era5_climo = lpfilt_era5_climo.rename({'latitude': 'lat', 'longitude': 'lon',
                                                  'doy_noleap': 'dayofyear'})

    return ds_era5, era5_climo, lpfilt_era5_climo


def preprocess_era5(ds):
    ds = ds.sel(latitude=slice(72, 23), longitude=slice(190, 295))
    ds = ds.resample(time='1D').mean('time')  # 6-hourly --> daily means
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))  # removes leap days
    doy = ds.time.dt.dayofyear
    if doy.max() == 366: doy = xr.where(doy > 60, doy - 1, doy)
    ds['doy_noleap'] = doy  # doy without leap days
    ds = ds.sortby('latitude')
    return ds


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
window = 30  ## (default 25)
cutoff = 1. / 30.  ## (default 1./11.)
wgts24 = low_pass_weights(window, cutoff)


def filter_wrapper(x):
    return sig.filtfilt(wgts24, np.sum(wgts24), x)


def run_harness(era5_path, save_path, save_path_climo, overwrite=False, unit='K', nmems=None):
    if nmems is None:
        nmems = len(ensmems)
    else:
        pass

    substr = "TREFHT"
    inserttxt = "-bc"
    inserttxt_climo = "-climo"
    temp_filenames_ = [os.path.basename(i) for i in natsorted(glob.glob(temp_path))]
    temp_filenames = [i[:i.index(substr) + len(substr)] + inserttxt + i[i.index(substr) + len(substr):] for i in
                      temp_filenames_]
    temp_climo_filenames = [i[:i.index(substr) + len(substr)] + inserttxt_climo + i[i.index(substr) + len(substr):] for
                            i in temp_filenames_]

    ds_era5, era5_climo, lpfilt_era5_climo = init_era5_dataset(era5_path)

    lat_era5 = ds_era5.latitude
    lon_era5 = ds_era5.longitude

    ds_out = xr.Dataset({'lat': (['lat'], lat_era5.data),
                         'lon': (['lon'], lon_era5.data), })

    for i, mem in tqdm(enumerate(ensmems[0:nmems])):
        temp_iter = xr.open_dataset(openme_temp[i])
        save_name = os.path.join(save_path, temp_filenames[i])
        save_name_climo = os.path.join(save_path_climo, temp_climo_filenames[i])

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

        if unit is 'C':
            temp_iter['TREFHT'].values = temp_iter['TREFHT'].values - 273.15
        elif unit is 'K':
            pass
            # wetbulb_iter['wetbulb'].values = wetbulb_iter['wetbulb'].values + 273.15
        else:
            print("You will find no support (i.e. love) for Farenheit here. Trying using 'C' for Celsius or 'K' for "
                  "Kelvin")

        d1 = cftime.DatetimeNoLeap(1979, 1, 1, 0, 0, 0)
        d2 = cftime.DatetimeNoLeap(2019, 12, 31, 11, 59, 59)
        cesm_climo = temp_iter['TREFHT'].sel(time=slice(d1, d2))
        cesm_climo = cesm_climo.groupby(cesm_climo.time.dt.dayofyear).mean()
        cesm_climo.to_netcdf(save_name_climo)
        cesm_climo.load()

        lpfilt_cesm_climo = xr.apply_ufunc(filter_wrapper, cesm_climo, input_core_dims=[['dayofyear']],
                                           output_core_dims=[['dayofyear']], vectorize=True)

        # regrid climo
        regridder = xesmf.Regridder(temp_iter, ds_out, 'bilinear')

        lpfilt_cesm_climo_re = regridder(lpfilt_cesm_climo.transpose('dayofyear', 'lat', 'lon'))

        # regrid CESM original to ERA5
        cesm_re = regridder(temp_iter)

        # calculate bias between one ensemble member and ERA5
        cesm_bias = lpfilt_cesm_climo_re - lpfilt_era5_climo

        cesm_bias_removed = cesm_re.groupby(cesm_re.time.dt.dayofyear) - cesm_bias

        cesm_bias_removed = cesm_bias_removed.astype('float32')
        cesm_bias_removed.to_netcdf(save_name)


run_harness('/adhara_b/ERA5/daily/T2m/', '/adhara_a/ljusten/TREFHT-bc/', '/adhara_a/ljusten/TREFHT-climatology/',
            overwrite=True)

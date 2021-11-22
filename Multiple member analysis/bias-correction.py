import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cpf
import glob
from natsort import natsorted
import numpy as np
import calendar
import xesmf
from matplotlib.gridspec import GridSpec
import scipy.signal as sig

wetbulb_path = '/adhara_a/ljusten/Wetbulb/*.nc'
temp_path = '/adhara_a/ljusten/TREFHT/*.nc'
era5_path = '/adhara_b/ERA5/daily/T2m/'

ensmems = list(range(1, 36, 1)) + list(range(101, 106, 1))


def init_data(temp_path, wetbulb_path, unit='C', n_mems=None):
    if n_mems is None:
        n_mems = len(ensmems)
    else:
        pass

    openme_temp = natsorted(glob.glob(temp_path))
    openme_wetbulb = natsorted(glob.glob(wetbulb_path))

    temp = []
    wetbulb = []
    for i, mem in enumerate(ensmems[0:n_mems]):
        temp_iter = xr.open_dataset(openme_temp[i])
        wetbulb_iter = xr.open_dataset(openme_wetbulb[i])

        if unit is 'C':
            temp_iter['TREFHT'].values = temp_iter['TREFHT'].values - 273.15
        elif unit is 'K':
            wetbulb_iter['wetbulb'].values = wetbulb_iter['wetbulb'].values + 273.15
        else:
            print(
                "You will find no support (i.e. love) for Farenheit here. Trying using 'C' for Celsius or 'K' for Kelvin")

        temp.append(temp_iter)
        wetbulb.append(wetbulb_iter)

        temp_agg = xr.concat(temp, dim='M', compat='override', coords='minimal', data_vars=['TREFHT'])
        wetbulb_agg = xr.concat(wetbulb, dim='M', compat='override', coords='minimal', data_vars=['wetbulb'])

    return temp_agg, wetbulb_agg


temp, wetbulb = init_data(temp_path, wetbulb_path, unit='K', n_mems=2)


def preprocess_era5(ds):
    ds = ds.sel(latitude = slice(72,23), longitude=slice(190, 295))
    ds = ds.resample(time = '1D').mean('time')  #6-hourly --> daily means
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29))) #removes leap days
    doy = ds.time.dt.dayofyear
    if doy.max()==366: doy = xr.where(doy>60, doy-1, doy)
    ds['doy_noleap'] = doy #doy without leap days
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


def filter_wrapper(x):
    return sig.filtfilt(wgts24, np.sum(wgts24), x)


# specify the window length for filters
window = 30  ## (default 25)
cutoff = 1. / 30.  ## (default 1./11.)
wgts24 = low_pass_weights(window, cutoff)


def bias_correction(temp, era5_path):
    # init era5 dataset
    ds_era5 = xr.open_mfdataset(sorted(glob.glob(era5_path + 'era5_global_*.nc')), combine='by_coords',
                                coords='minimal', compat='override', preprocess=preprocess_era5)

    # Get day of the year means
    doym_era5 = dayofyearmean(ds_era5)['t2m']
    doym_era5.load()

    doym_cesm = temp['TREFHT'].groupby(temp['TREFHT'].time.dt.dayofyear).mean()
    doym_cesm.load()

    # Apply low pass Lanczos Filter
    doym_lpfilt_era5 = xr.apply_ufunc(filter_wrapper, doym_era5, input_core_dims=[['doy_noleap']],
                                      output_core_dims=[['doy_noleap']], vectorize=True)

    doym_lpfilt_cesm = xr.apply_ufunc(filter_wrapper, doym_cesm, input_core_dims=[['dayofyear']],
                                      output_core_dims=[['dayofyear']], vectorize=True)

    lat_era5 = ds_era5.latitude
    lon_era5 = ds_era5.longitude

    # Create empty dataset for the bias corrected values
    ds_out = xr.Dataset({'lat': (['lat'], lat_era5.data),
                         'lon': (['lon'], lon_era5.data), })

    # regrid climo
    regridder = xesmf.Regridder(temp, ds_out, 'bilinear')
    doym_lpfilt_cesm_re = regridder(doym_lpfilt_cesm.transpose('dayofyear', 'lat', 'lon', 'M'))

    # regrid CESM original to ERA5
    cesm_re = regridder(temp)

    # make sure ERA5 dims have same names as CESMLE dims
    doym_lpfilt_era5 = doym_lpfilt_era5.rename({'latitude': 'lat', 'longitude': 'lon', 'doy_noleap': 'dayofyear'})

    # calculate bias between one ensemble member and ERA5
    cesm_bias = doym_lpfilt_cesm_re - doym_lpfilt_era5

    cesm_bias_removed = cesm_re.groupby(cesm_re.time.dt.dayofyear) - cesm_bias
    return cesm_bias_removed


temp_corrected = bias_correction(temp)

# Description: refactor the files from emaroon such that temp and qbot are represented by single files for each
# ensemble. Additionally, restrict the temporal and spatial domain of the refactored files with the config dict.
# The script also prepares the pressure file by taking the time mean over the press_path dataset.

# Author: Lenni Justen
# Date: 11/10/21

from tqdm import tqdm
import xarray as xr
import glob
import cftime
import os

press_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/monthly/PS/b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h0.PS.185001-200512.nc'
trefht_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/TREFHT/'
qbot_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/QBOT/'
grid_path = '/home1/ljusten/wetbulb_metpy_lowres_2_15_grid.pkl'

trefht_save_path = '/adhara_a/ljusten/TREFHT/'
qbot_save_path = '/adhara_a/ljusten/QBOT/'

ensmems = list(range(1, 36, 1)) + list(range(101, 106, 1))


config = {
    'start': cftime.DatetimeNoLeap(1920, 1, 1, 0, 0, 0),
    'end': cftime.DatetimeNoLeap(2100, 12, 31, 0, 0, 0),
    'lat': (23, 72),
    'lon': (190, 295),
}


def preprocess(ds):
    ds_t = ds.sel(time=slice(config['start'], config['end']))
    ds_g = ds_t.sel(lat=slice(config['lat'][0], config['lat'][1])).sel(lon=slice(config['lon'][0], config['lon'][1]))
    return ds_g


def refactor_dataset(path, data_vars, overwrite=False):
    openme = [sorted(glob.glob(path + 'b.e11.B[2,R]*.' + str(ii).zfill(3) + '*.nc')) for ii in ensmems]

    for mem, oo in tqdm(enumerate(openme)):
        save_name = '/adhara_a/ljusten/{}/b.e11.BRCP85C5CNBDRD.f09_g16.{}.cam.h1.{}.19200101-21001231.nc'.format(
            data_vars[0], ensmems[mem], data_vars[0])

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

        try:
            ds = xr.open_mfdataset(oo, combine='by_coords', compat='override', coords='minimal', data_vars=data_vars,
                                   preprocess=preprocess, chunks={})

            ds.to_netcdf(save_name)
        except:
            print('WARNING: Ensemble member {} could not be concatenated. Trying naive concat method...'.format(mem))

            ds_ = []
            for f in openme[mem]:
                ds_.append(xr.open_dataset(f))

            ds = xr.concat(ds_, dim='time', compat='override', coords='minimal', data_vars=data_vars)
            ds1 = ds.sel(lat=slice(config['lat'][0], config['lat'][1])).sel(
                lon=slice(config['lon'][0], config['lon'][1]))

            ds1.to_netcdf(save_name)


refactor_dataset(qbot_path, ['QBOT'])
refactor_dataset(trefht_path, ['TREFHT'])


def single_file_save(files, data_var):
    ds = []
    for f in tqdm(files):
        ds.append(xr.open_dataset(f))

    return xr.concat(ds, dim='M', data_vars=[data_var], compat='override', coords='minimal')


# ds_temp = single_file_save(glob.glob(os.path.join(trefht_save_path,'*.nc')), 'TREFHT')
# ds_temp.drop(['cosp_sr', 'cosp_prs', 'cosp_tau', 'cosp_ht','cosp_tau_modis', 'cosp_htmisr', 'cosp_scol', 'cosp_sza'])
# ds_temp.to_netcdf('/adhara_a/ljusten/b.e11.BRCP85C5CNBDRD.f09_g16.ALL.cam.h1.TREFHT.19200101-21001231.nc')
#
#
# ds_qbot = single_file_save(glob.glob(os.path.join(qbot_save_path,'*.nc')), 'QBOT')
# ds_qbot.drop(['cosp_sr', 'cosp_prs', 'cosp_tau', 'cosp_ht','cosp_tau_modis', 'cosp_htmisr', 'cosp_scol', 'cosp_sza'])
# ds_qbot.to_netcdf('/adhara_a/ljusten/b.e11.BRCP85C5CNBDRD.f09_g16.ALL.cam.h1.QBOT.19200101-21001231.nc')


def pressure_prep(path):
    press = xr.open_dataset(path)

    press_ = preprocess(press)

    press_mean = press_.mean('time')
    press_mean['PS'] = press_mean.PS.assign_attrs(units='Pa')
    return press_mean


ds_press = pressure_prep(press_path)
ds_press.to_netcdf('/adhara_a/ljusten/b.e11.B20TRC5CNBDRD.f09_g16.MEAN-001.cam.h0.PS.185001-200512.nc')

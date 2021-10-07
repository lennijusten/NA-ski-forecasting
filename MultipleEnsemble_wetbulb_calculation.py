import numpy as np
import os
import metpy.calc
import pickle
import xarray as xr
import pandas as pd

press_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/monthly/PS/b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h0.PS.185001-200512.nc'
trefht_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/TREFHT/'
qbot_path = '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/QBOT/'
grid_path = '/home1/ljusten/wetbulb_metpy_lowres_2_15_grid.pkl'


def select_grid(temp, qbot, press, bot_lat, top_lat, left_lon, right_lon):
    ds_temp = temp.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_qbot = qbot.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_press = press.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    return ds_temp, ds_qbot, ds_press


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


def find_nearest(array, value):  # Returns index of element in array nearest to input value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def select_time(temp, qbot, start, end):
    ds_temp = temp.sel(time=slice(start, end))
    ds_qbot = qbot.sel(time=slice(start, end))
    return ds_temp, ds_qbot


def save_data(trefht_path, qbot_path, press_path, grid_path, wetbulb_save_path='/home1/ljusten/CESM/Wetbulb/',
              temp_save_path='/home1/ljusten/CESM/Temp/'):  # pass as lists
    with (open(grid_path, "rb")) as openfile:  # Load reference grid
        while True:
            try:
                wetbulb_key = pickle.load(openfile)
            except EOFError:
                break

    wetbulb_key = np.array(wetbulb_key)

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

    members = [[t, q, t.split('f09_g16.', 1)[1][0:3]] for t, q in zip(os.listdir(trefht_path), os.listdir(qbot_path))]

    df = pd.DataFrame(members, columns=[['t_fname', 'q_fname', 'ensembleID']])
    df.columns = list(map(''.join, df.columns.values))
    df.sort_values(by='ensembleID', inplace=True)

    # 23, 72, 190, 295
    press = xr.open_dataset(press_path)
    press1 = press.sel(lat=slice(23, 72)).sel(lon=slice(190, 295))
    press_mean = press1.mean('time')
    press_mean['PS'] = press_mean.PS.assign_attrs(units='Pa')

    for ID in pd.unique(df['ensembleID']):
        t_files = df['t_fname'][df['ensembleID'] == ID].values
        q_files = df['q_fname'][df['ensembleID'] == ID].values

        temp_ds = []
        qbot_ds = []
        for t, q in zip(t_files, q_files):
            print('t = ', t)
            print('qbot = ', q)

            temp_ = xr.open_dataset(os.path.join(trefht_path, t))
            qbot_ = xr.open_dataset(os.path.join(qbot_path, q))
            temp_ds.append(temp_)
            qbot_ds.append(qbot_)

        temp = xr.concat(temp_ds, dim="time")
        qbot = xr.concat(qbot_ds, dim="time")

        temp1, qbot1, _ = select_grid(temp, qbot, press, 23, 72, 190, 295)  # Standard NA grid
        temp2, qbot2 = select_time(temp1, qbot1, temp1['time'][0].values, temp1['time'][-1].values)

        print(qbot2)
        print(qbot2['QBOT'])

        RH, dewpoint = get_dewpoint(temp2, press_mean, qbot2, 'RH')

        temp_degC = temp2['TREFHT'].values - 273.15  # Convert temp to degC (no metpy units)

        # Flatten all wetbulb inputs
        long_temp = temp_degC.flatten()
        long_press = np.tile(press_mean['PS'].values.flatten(), temp_degC.shape[0])
        long_dewpoint = dewpoint.values.flatten()

        wetbulb_search = []
        for i in range(len(long_temp)):
            p_idx = find_nearest(p_space, long_press[i])
            t_idx = find_nearest(t_space, long_temp[i])
            dp_idx = find_nearest(dp_space, long_dewpoint[i])
            wetbulb_search.append(
                wetbulb_key[p_idx][dp_idx][t_idx])  # append element of reference grid nearest too inputs

        wetbulb = np.reshape(np.array(wetbulb_search), temp_degC.shape)  # reshape flattened wetbulb into original dims

        wb = xr.DataArray(wetbulb, dims=['time', 'lat', 'lon'],
                          coords={'time': temp2.time, 'lat': temp2.lat, 'lon': temp2.lon})
        wb.attrs['units'] = 'degC'
        wb.attrs['long_name'] = 'Wetbulb temperature'

        print(wb)

        try:
            wb = wb.rename_vars({"__xarray_dataarray_variable__": "wetbulb"})
        except:
            pass

        # wb = wb.rename_vars({"__xarray_dataarray_variable__": "wetbulb"})

        wetbulb_spath = wetbulb_save_path + 'b.e11.BRCP85C5CNBDRD.f09_g16.{}.cam.h1.WETBULB.19200101-21001231'.format(
            ID)
        temp_spath = temp_save_path + 'b.e11.B20TRC5-BRCP85C5CNBDRD.f09_g16.{}.cam.h1.TREFHT.19200101-21001231.nc'.format(
            ID)

        wb.to_netcdf(wetbulb_spath)
        temp2.to_netcdf(temp_spath)


save_data(trefht_path, qbot_path, '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/monthly/PS/b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h0.PS.185001-200512.nc',
          grid_path, wetbulb_save_path='/home1/ljusten/CESM/Wetbulb/', temp_save_path='/home1/ljusten/CESM/Temp/')


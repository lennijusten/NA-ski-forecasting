import numpy as np
import pandas as pd
import metpy.calc
import xarray as xr
import matplotlib.pyplot as plt


def init_data():
    ds_press = xr.open_dataset(
        '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/monthly/PS/b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h0.PS.185001-200512.nc')
    ds_temp = xr.open_dataset(
        '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/TREFHT/b.e11.BRCP85C5CNBDRD.f09_g16.001.cam.h1.TREFHT.20810101-21001231.nc')
    ds_qbot = xr.open_dataset(
        '/adhara_a/emaroon/cesmLE1/atm/proc/tseries/daily/QBOT/b.e11.BRCP85C5CNBDRD.f09_g16.001.cam.h1.QBOT.20810101-21001231.nc')

    return ds_temp, ds_press, ds_qbot


temp, press, qbot = init_data()


def select_grid(temp, press, qbot, bot_lat, top_lat, left_lon, right_lon):
    ds_temp = temp.sel(lat=slice(bot_lat,top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_press = press.sel(lat=slice(bot_lat,top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_qbot = qbot.sel(lat=slice(bot_lat,top_lat)).sel(lon=slice(left_lon, right_lon))
    return temp, press, qbot

temp1, press1, qbot1 = select_grid(temp, press, qbot,-25,-20,130,140)

press_mean = press1.mean('time')
press_mean['PS'] = press_mean.PS.assign_attrs(units='Pa')


def select_time(temp, qbot, start, end):
    ds_temp = temp.sel(time=slice(start, end))
    ds_qbot = qbot.sel(time=slice(start, end))
    return temp, qbot

temp2, qbot2 = select_time(temp1, qbot1, '2081-01-01 00:00:00', '2082-01-01 00:00:00')


def get_dewpoint(temp,press,qbot,method):
    if method == 'RH':
        RH = metpy.calc.relative_humidity_from_mixing_ratio(qbot['QBOT'], temp['TREFHT'], press['PS'])
        dewpoint = metpy.calc.dewpoint_from_relative_humidity(temp['TREFHT'], RH)
        return RH, dewpoint
    elif method == 'WVPP':
        vapor_pressure = metpy.calc.vapor_pressure(press['PS'], qbot['QBOT'])  # https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.vapor_pressure.html
        dewpoint = metpy.calc.dewpoint(vapor_pressure)
        return vapor_pressure, dewpoint
    else:
        print("Unknown method--Relative Humidity: 'RH', Water Vapor Partial Pressure: 'WVPP'")


# RH, dewpointA = get_dewpoint(temp2, press_mean, qbot2, 'RH')

vapor_pressure, dewpointB = get_dewpoint(temp2, press_mean, qbot2, 'WVPP')

wetbulb = metpy.calc.wet_bulb_temperature(press_mean['PS'], temp2['TREFHT'], dewpointB)
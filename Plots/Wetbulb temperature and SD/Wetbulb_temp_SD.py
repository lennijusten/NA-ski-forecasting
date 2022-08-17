import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cpf
import cartopy.feature as cfeature
import glob
from natsort import natsorted
import numpy as np
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import cmaps

wetbulb_path = '/adhara_a/ljusten/Wetbulb-bc/*.nc'
temp_path = '/adhara_a/ljusten/TREFHT-bc-degC/*.nc'
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
    for i, mem in tqdm(enumerate(ensmems[0:n_mems])):
        temp_iter = xr.open_dataset(openme_temp[i], chunks={})
        wetbulb_iter = xr.open_dataset(openme_wetbulb[i], chunks={})

        if unit is 'C':
            pass
        elif unit is 'K':
            print('This version is slow and not yet fully optimized')
            temp_iter['TREFHT'] = temp_iter['TREFHT'] + 273.15
            wetbulb_iter['wetbulb'] = wetbulb_iter['wetbulb'] + 273.15
        else:
            print(
                "You will find no support (i.e. love) for Farenheit here. Trying using 'C' for Celsius or 'K' for Kelvin")

        temp.append(temp_iter)
        wetbulb.append(wetbulb_iter)

    temp_agg = xr.concat(temp, dim='M', compat='override', coords='minimal', data_vars=['TREFHT'])
    wetbulb_agg = xr.concat(wetbulb, dim='M', compat='override', coords='minimal', data_vars=['wetbulb'])

    return temp_agg, wetbulb_agg


temp, wetbulb = init_data(temp_path, wetbulb_path, unit='C', n_mems=40)

# Inputs
snow_thresh = -2.0
months = [11,12,1,2,3] # if months = none, add full year
periods = [2030,2060,2090]

ref_period = ('1980-01-01 00:00:00', '2000-01-01 00:00:00')

df = pd.read_csv('ski_locations_w_snowmaking_days_extended.csv', index_col =0)
decades = [int(y) for y in np.unique(wetbulb['time.year']) if y%10==0]
decades_labels = [int(y) if y%20==0 else '' for y in decades]


def select_time(var, start, end):
    return var.sel(time=slice(start, end))


def rect_from_bound(xmin, xmax, ymin, ymax):
    """Returns list of (x,y)'s for a rectangle"""
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]


def make_geometries(projection=ccrs.LambertConformal()):
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    datacoord = ccrs.PlateCarree()

    # request data for use by geopandas
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'

    shpfilename = shapereader.natural_earth(resolution, category, name)
    df = gpd.read_file(shpfilename)

    # get geometry of a country
    poly = [df.loc[df['ADMIN'] == 'United States of America']['geometry'].values[0]]
    poly2 = [df.loc[df['ADMIN'] == 'Canada']['geometry'].values[0]]

    pad1 = 0.1  # padding, degrees unit
    exts = [poly[0].bounds[0] - pad1, poly[0].bounds[2] + pad1, poly[0].bounds[1] - pad1, poly[0].bounds[3] + pad1];

    # make a mask polygon by polygon's difference operation
    msk = Polygon(rect_from_bound(*exts)).difference(poly[0].simplify(0.01))
    msk_pc = projection.project_geometry(msk, datacoord)  # project geometry to the projection
    return states_provinces, poly, poly2, msk_pc


states_provinces, poly, poly2, msk_pc = make_geometries()


#  # create fig and axes using intended projection
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
# ax.add_geometries(poly2, crs=ccrs.PlateCarree(), facecolor='black', edgecolor='black')

def temp_and_SD(temp, wetbulb, df, thresh, periods, months=None, projection=ccrs.LambertConformal(),
               extent=[-130, -65, 24, 50], reference_period=None):
    datacoord = ccrs.PlateCarree()

    # Setup figure
    fig, axes = plt.subplots(nrows=len(periods), ncols=2, figsize=(15, 15), subplot_kw={'projection': projection})
    fig.subplots_adjust(wspace=0, hspace=0)

    # Coloumn 1
    for i, p in enumerate(periods):
        wetbulb_iter = wetbulb.sel(time=wetbulb.time.dt.month.isin(months))
        temp_iter = temp.sel(time=temp.time.dt.month.isin(months))

        wetbulb_pmean = wetbulb_iter.groupby('time.year')[p].mean('time').mean('M')
        temp_pmean = temp_iter.groupby('time.year')[p].mean('time').mean('M')
        temp_sub0 = temp_pmean <= 0.0
        wetbulb_pstd = wetbulb_iter.groupby('time.year')[p].mean('time').std('M')

        # setup projection and axis properties
        axes[i, 0].set_extent(extent, crs=datacoord)  # set lat,lon borders of plot
        axes[i, 0].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)  # mask oceans
        axes[i, 0].add_feature(states_provinces, edgecolor='dimgray', zorder=2)
        axes[i, 0].coastlines(zorder=4, linewidth=2)
        axes[i, 0].add_feature(cfeature.BORDERS, zorder=5)
        axes[i, 0].add_geometries(poly2, crs=projection, facecolor='white', edgecolor='none', zorder=3)
        axes[i, 0].add_geometries(poly, crs=projection, facecolor='none', edgecolor='black', zorder=4)
        axes[i, 0].add_geometries(msk_pc, projection, zorder=4, facecolor='white', edgecolor='none', alpha=1)
        axes[i, 0].axis('off')

        axes[i, 1].set_extent(extent, crs=datacoord)  # set lat,lon borders of plot
        axes[i, 1].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)  # mask oceans
        axes[i, 1].add_feature(states_provinces, edgecolor='dimgray', zorder=2)
        axes[i, 1].coastlines(zorder=4, linewidth=2)
        axes[i, 1].add_feature(cfeature.BORDERS, zorder=5)
        axes[i, 1].add_geometries(poly2, crs=projection, facecolor='white', edgecolor='none', zorder=3)
        axes[i, 1].add_geometries(poly, crs=projection, facecolor='none', edgecolor='black', zorder=4)
        axes[i, 1].add_geometries(msk_pc, projection, zorder=4, facecolor='white', edgecolor='none', alpha=1)
        axes[i, 1].axis('off')

        # Add ylabels
        axes[i, 0].text(0.08, 0.55, '{}'.format(p), va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=axes[i, 0].transAxes, zorder=5)

        if reference_period is None:
            axes[0, 0].set_title('Ensemble Mean of Wet-bulb Temperature')
            axes[0, 1].set_title('Ensemble Standard Deviation of Wet-bulb Temperature')
            wb_min = -12
            wb_max = 14
            wb_labels = np.linspace(wb_min, wb_max, 14)
            cmap1 = cmaps.amwg_blueyellowred

        else:
            axes[0, 0].set_title('Ensemble Mean Wet-bulb Temperature (1980-2000 ref period)')
            axes[0, 1].set_title('Ensemble Standard Deviation of Wet-bulb Temperature')

            wb_min = 0
            wb_max = 8
            wb_labels = np.linspace(wb_min, wb_max, 9)
            cmap1 = cm.hot_r

            wetbulb_ref = select_time(wetbulb_iter, reference_period[0], reference_period[1]).mean('time').mean('M')
            wetbulb_pmean = wetbulb_pmean - wetbulb_ref

        w = wetbulb_pmean['wetbulb'].plot.contourf(transform=datacoord,
                                                   ax=axes[i, 0], cmap=cmap1, vmin=wb_min, vmax=wb_max,
                                                   add_colorbar=False, levels=wb_labels, add_labels=False)

        sub0 = temp_sub0['TREFHT'].plot.contour(transform=datacoord, ax=axes[i,0], colors='k', linewidths=0.5)

        w_std = wetbulb_pstd['wetbulb'].plot.contourf(transform=datacoord,
                                                      ax=axes[i, 1], cmap=cm.viridis, vmin=0, vmax=2,
                                                      add_colorbar=False, levels=np.linspace(0, 2, 5), add_labels=False)


    plt.savefig('wetbulb_temp_and_SD.png', dpi=300)

    fig, ax = plt.subplots()
    ax.axis('off')
    fig.colorbar(w, ax=ax, label="Wet-bulb mean ({}C)".format(u'\N{DEGREE SIGN}'), location='top')
    plt.savefig('wetbulb_temp_colorbar.png', dpi=300)

    fig, ax = plt.subplots()
    ax.axis('off')
    fig.colorbar(w_std, ax=ax, label="Wet-bulb standard deviation ({}C)".format(u'\N{DEGREE SIGN}'), location='top')
    plt.savefig('wetbulb_SD_colorbar.png', dpi=300)


temp_and_SD(temp, wetbulb, df, snow_thresh, periods, months, reference_period=ref_period)
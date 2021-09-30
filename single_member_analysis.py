import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cpf

wetbulb_path = '/Users/lenni/Downloads/b.e11.BRCP85C5CNBDRD.f09_g16.001.cam.h1.WETBULB.19200101-21001231.nc'
temp_path = '/Users/lenni/Downloads/b.e11.B20TRC5-BRCP85C5CNBDRD.f09_g16.001.cam.h1.TREFHT.19200101-21001231.nc'


def init_data(wetbulb_path, temp_path):
    wetbulb = xr.open_dataset(wetbulb_path)
    wetbulb = wetbulb.rename_vars({"__xarray_dataarray_variable__": "wetbulb"})

    temp = xr.open_dataset(temp_path)
    # temp = xr.concat(temp_list, dim="time")
    temp['TREFHT'] = temp['TREFHT'] - 273.15
    return wetbulb, temp


wetbulb, temp = init_data(wetbulb_path, temp_path)


def select_time(xarray, start, end):
    return xarray.sel(time=slice(start, end))


temp_ = select_time(temp, '1920-01-01 00:00:00', '2100-12-31 00:00:00')


def select_grid(wetbulb, temp, bot_lat, top_lat, left_lon, right_lon):
    ds_temp = temp.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    ds_wetbulb = wetbulb.sel(lat=slice(bot_lat, top_lat)).sel(lon=slice(left_lon, right_lon))
    return ds_wetbulb, ds_temp


wetbulb1, temp1 = select_grid(wetbulb, temp, 23, 72, 190, 295)  # Standard NA grid
residual = temp1['TREFHT'] - wetbulb1['wetbulb']

ref_period = ('1980-01-01 00:00:00', '2000-01-01 00:00:00')


def reference_period_seasonal_mean(var, start, end):
    # Returns season mean of xarray over specified reference period
    var_ref = select_time(var, start, end)
    return var_ref.groupby('time.season').mean('time')


def seasonal_averages(temp, wetbulb, reference_period=None, drop_reference_period=None):
    residual = temp['TREFHT'] - wetbulb['wetbulb']

    if reference_period is None:
        # return seasonal wetbulb,temp,residual means over the entire period of the input data
        wetbulb_season = wetbulb.groupby('time.season').mean('time')
        temp_season = temp.groupby('time.season').mean('time')
        residual_season = residual.groupby('time.season').mean('time')
        return temp_season, wetbulb_season, residual_season

    elif reference_period is not None and drop_reference_period is False:
        # return seasonal wetbulb,temp,residual means over the entire period of input data - mean from reference period
        wetbulb_season = wetbulb.groupby('time.season').mean('time')
        temp_season = temp.groupby('time.season').mean('time')

        wetbulb_season_ref = reference_period_seasonal_mean(wetbulb, reference_period[0], reference_period[1])
        temp_season_ref = reference_period_seasonal_mean(temp, reference_period[0], reference_period[1])

        temp_season_diff = temp_season - temp_season_ref
        wetbulb_season_diff = wetbulb_season - wetbulb_season_ref
        residual_season = temp_season_diff['TREFHT'] - wetbulb_season_diff['wetbulb']
        return temp_season_diff, wetbulb_season_diff, residual_season

    elif reference_period is not None and drop_reference_period is True:
        # return seasonal wetbulb,temp,residual means from the end of the reference period to the end of the input
        # data - mean from the reference period
        temp_season_ref = reference_period_seasonal_mean(temp, reference_period[0], reference_period[1])
        wetbulb_season_ref = reference_period_seasonal_mean(wetbulb, reference_period[0], reference_period[1])

        temp_dropref = select_time(temp, reference_period[1], temp.time[-1])
        wetbulb_dropref = select_time(wetbulb, reference_period[1], wetbulb.time[-1])

        wetbulb_dropref_season = wetbulb_dropref.groupby('time.season').mean('time')
        temp_dropref_season = temp_dropref.groupby('time.season').mean('time')

        temp_season_diff = temp_dropref_season - temp_season_ref
        wetbulb_season_diff = wetbulb_dropref_season - wetbulb_season_ref
        residual_season = temp_season_diff['TREFHT'] - wetbulb_season_diff['wetbulb']
        return temp_season_diff, wetbulb_season_diff, residual_season

    else:
        print('Error: drop_reference_period must be either True or False if reference_period is not None')


temp_season, wetbulb_season, residual_season = seasonal_averages(temp1, wetbulb1, reference_period=ref_period,
                                                                 drop_reference_period=True)


def seasonal_average_plot(projection=ccrs.PlateCarree(), extent=[-170, -65, 23.08900524, 71.15183246],
                          ref_period=False):
    if ref_period:
        temp_min = -10
        temp_max = abs(temp_min)
        res_min = -5
        res_max = abs(res_min)
    else:
        temp_min = -30
        temp_max = abs(temp_min)
        res_min = -15
        res_max = abs(res_min)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    # Create subplot over all other subplots to display common ylabel
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)  # hide tick and tick label of the big axes
    plt.grid(False)
    plt.ylabel('Season', labelpad=35)  # Use labelpad to push ylabel back

    # Removes whitespace
    for ax in axes.flat:
        ax.axes.axis('tight')
        ax.set_xlabel('')

    # Loop through seasons and plot data
    for i, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
        axes[i, 0].coastlines(zorder=2)  # zorder=2 > zorder=1 so the coastlines will be plotted over the ocean mask
        axes[i, 1].coastlines(zorder=2)
        axes[i, 2].coastlines(zorder=2)

        axes[i, 0].set_extent(extent, crs=projection)  # set lat,lon borders of plot
        axes[i, 1].set_extent(extent, crs=projection)
        axes[i, 2].set_extent(extent, crs=projection)

        axes[i, 0].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)  # mask oceans
        axes[i, 1].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)
        axes[i, 2].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)

        # Plot colormaps from xarray
        t = temp_season['TREFHT'].sel(season=season).plot.pcolormesh(
            ax=axes[i, 0], vmin=temp_min, vmax=temp_max, cmap='bwr',
            add_colorbar=False, extend='both')

        w = wetbulb_season['wetbulb'].sel(season=season).plot.pcolormesh(
            ax=axes[i, 1], vmin=temp_min, vmax=temp_max, cmap='bwr',
            add_colorbar=False, extend='both')

        r = residual_season.sel(season=season).plot.pcolormesh(
            ax=axes[i, 2], vmin=res_min, vmax=res_max, cmap='coolwarm',
            add_colorbar=False)

        # Add grid and gridlabels to the bottom and left side of axes
        grid_t = axes[i, 0].gridlines(crs=projection, draw_labels=True)
        grid_t.top_labels = False
        grid_t.right_labels = False
        grid_w = axes[i, 1].gridlines(crs=projection, draw_labels=True)
        grid_w.top_labels = False
        grid_w.right_labels = False
        grid_w.left_labels = False
        grid_r = axes[i, 2].gridlines(crs=projection, draw_labels=True)
        grid_r.top_labels = False
        grid_r.right_labels = False

        # Add season ylabels since ax.set_ylabel is broken in cartopy: https://stackoverflow.com/questions/35479508/cartopy-set-xlabel-set-ylabel-not-ticklabels
        axes[i, 0].text(-0.2, 0.55, season, va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=axes[i, 0].transAxes)

        #         # Plots colorbars on each row of the second column
        #         fig.colorbar(t, ax=axes[i,0:2], location='right',extend='both',label='degC')
        #         fig.colorbar(r, ax=axes[i,2],extend='both',label='degC')

        # Set all titles to blank
        axes[i, 0].set_title('')
        axes[i, 1].set_title('')
        axes[i, 2].set_title('')

    # Plots one long colorbar across all rows on the second column
    fig.colorbar(t, ax=axes[:, 0:2], location='right', extend='both', label='degC')
    fig.colorbar(r, ax=axes[:, 2], location='right', label='degC')

    # Add titles to top of columns
    axes[0, 0].set_title('Drybulb Temperature')
    axes[0, 1].set_title('Wetbulb Temperature')
    axes[0, 2].set_title('Drybulb - Wetbulb Temperature')

    fig.suptitle('Average Seasonal Wetbulb and Drybulb Temperature over North America (1920-2100)', fontsize=16, y=1.02)
    plt.savefig('seasonal-comparison-wetbulb.png', dpi=300)


seasonal_average_plot(ref_period=True)
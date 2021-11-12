import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cpf
import glob
from natsort import natsorted
import calendar
import numpy as np

wetbulb_path = '/adhara_a/ljusten/Wetbulb/*.nc'
temp_path = '/adhara_a/ljusten/TREFHT/*.nc'

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
                "You will find support (i.e. love) for Farenheit here. Trying using 'C' for Celsius or 'K' for Kelvin")

        temp.append(temp_iter)
        wetbulb.append(wetbulb_iter)

        temp_agg = xr.concat(temp, dim='M', compat='override', coords='minimal', data_vars=['TREFHT'])
        wetbulb_agg = xr.concat(wetbulb, dim='M', compat='override', coords='minimal', data_vars=['wetbulb'])

    return temp_agg, wetbulb_agg


temp, wetbulb = init_data(temp_path, wetbulb_path, unit='C', n_mems=3)


def select_time(var, start, end):
    return var.sel(time=slice(start, end))


def residual_func(temp, wetbulb):
    return [t_mem['TREFHT']-w_mem['wetbulb'] for t_mem,w_mem in zip(temp, wetbulb)]


def mean_ensemble(ensemble, data_var):
    return ensemble[data_var].mean(dim='M')


def std_ensemble(ensemble, data_var):
    return ensemble[data_var].std(dim='M')


periods = [['2010-01-01 00:00:00', '2040-01-01 00:00:00'], ['2040-01-01 00:00:00', '2070-01-01 00:00:00'], ['2070-01-01 00:00:00', '2100-12-31 00:00:00']]

ref_period = ('1980-01-01 00:00:00', '2010-01-01 00:00:00')


def time_slice_plot(temp, wetbulb, season='DJF', projection=ccrs.PlateCarree(), extent=[-170, -65, 23.08900524, 71.15183246],
                    reference_period=None, include_residual=False):

    if reference_period is None:
        temp_min = -40
        temp_max = abs(temp_min)
        res_min = -10
        res_max = abs(res_min)
    else:
        temp_min = -15
        temp_max = abs(temp_min)
        res_min = -5
        res_max = abs(res_min)

    if include_residual:
        fig, axes = plt.subplots(nrows=len(periods), ncols=3, figsize=(16, 12),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(nrows=len(periods), ncols=2, figsize=(12, 12),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

    for ax in axes.flat:
        ax.axes.axis('tight')
        ax.set_xlabel('')

    for i, p in enumerate(periods):

        axes[i, 0].coastlines(zorder=2)  # zorder=2 > zorder=1 so the coastlines will be plotted over the ocean mask
        axes[i, 1].coastlines(zorder=2)

        axes[i, 0].set_extent(extent, crs=projection)  # set lat,lon borders of plot
        axes[i, 1].set_extent(extent, crs=projection)

        axes[i, 0].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)  # mask oceans
        axes[i, 1].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)

        if include_residual:
            axes[i, 2].coastlines(zorder=2)
            axes[i, 2].set_extent(extent, crs=projection)
            axes[i, 2].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)
        else:
            pass

        # select wetbulb and temperature from period start and end
        wetbulb_p = select_time(wetbulb, p[0], p[1])
        temp_p = select_time(temp, p[0], p[1])

        if reference_period is None:
            temp_pmean = temp_p.groupby('time.season').mean('time')
            wetbulb_pmean = wetbulb_p.groupby('time.season').mean('time')
            residual = temp_pmean - wetbulb_pmean
        else:
            # subtract reference period mean from period means of temp
            temp_ref = select_time(temp, reference_period[0], reference_period[1]).groupby('time.season').mean('time')
            wetbulb_ref = select_time(wetbulb, reference_period[0], reference_period[1]).groupby('time.season').mean(
                'time')

            temp_pmean = temp_p.groupby('time.season').mean('time') - temp_ref
            wetbulb_pmean = wetbulb_p.groupby('time.season').mean('time') - wetbulb_ref
            residual = temp_pmean - wetbulb_pmean

        # Plot colormaps from xarray
        t = temp_pmean.sel(season=season).plot.pcolormesh(
            ax=axes[i, 0], vmin=temp_min, vmax=temp_max, cmap='bwr',
            add_colorbar=False, extend='both')

        w = wetbulb_pmean.sel(season=season).plot.pcolormesh(
            ax=axes[i, 1], vmin=temp_min, vmax=temp_max, cmap='bwr',
            add_colorbar=False, extend='both')

        if include_residual:
            r = residual.sel(season=season).plot.pcolormesh(
                ax=axes[i, 2], vmin=res_min, vmax=res_max, cmap='bwr',
                add_colorbar=False, extend='both')

            grid_r = axes[i, 2].gridlines(crs=projection, draw_labels=True)
            grid_r.top_labels = False
            grid_r.right_labels = False

            axes[0, 2].set_title('Drybulb - Wetbulb Temperature')
            fig.colorbar(r, ax=axes[:, 2], location='right', extend='both', label='degC')

        else:
            pass

        # Add grid and gridlabels to the bottom and left side of axes
        grid_t = axes[i, 0].gridlines(crs=projection, draw_labels=True)
        grid_t.top_labels = False
        grid_t.right_labels = False
        grid_w = axes[i, 1].gridlines(crs=projection, draw_labels=True)
        grid_w.top_labels = False
        grid_w.right_labels = False
        grid_w.left_labels = False

        # Add season ylabels since ax.set_ylabel is broken in cartopy: https://stackoverflow.com/questions/35479508/cartopy-set-xlabel-set-ylabel-not-ticklabels
        axes[i, 0].text(-0.3, 0.55, '{}-{}'.format(p[0][:4], p[1][:4]), va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=axes[i, 0].transAxes)

        # # Plots colorbars on each row of the second column
        # fig.colorbar(t, ax=axes[i,0:2], location='right',extend='both',label='degC')
        # fig.colorbar(r, ax=axes[i,2],extend='both',label='degC')

        # Set all titles to blank
        axes[i, 0].set_title('')
        axes[i, 1].set_title('')

    # Plots one long colorbar across all rows on the second column
    fig.colorbar(t, ax=axes[:, 0:2], location='right', extend='both', label='degC')

    # Add titles to top of columns
    axes[0, 0].set_title('Drybulb Temperature')
    axes[0, 1].set_title('Wetbulb Temperature')

    fig.suptitle('Time evolution of {} temperatures over NA'.format(season), fontsize=16, y=1.02)
    plt.savefig('winter-comparison-wetbulb.png', dpi=300)


time_slice_plot(mean_ensemble(temp, 'TREFHT'), mean_ensemble(wetbulb, 'wetbulb'), season='DJF',
                reference_period=ref_period)

months = [11, 12, 1, 2, 3]

ref_period = ('1980-01-01 00:00:00', '2010-01-01 00:00:00')
display_period = ('2020-01-01 00:00:00', '2030-01-01 00:00:00')


def monthly_comparison_plot(temp, wetbulb, period, months=months, projection=ccrs.PlateCarree(),
                            extent=[-170, -65, 23.08900524, 71.15183246], reference_period=None,
                            include_residual=False):
    if reference_period is not None:
        temp_min = -10
        temp_max = abs(temp_min)
        res_min = -5
        res_max = abs(res_min)
    else:
        temp_min = -40
        temp_max = abs(temp_min)
        res_min = -10
        res_max = abs(res_min)

    if include_residual:
        fig, axes = plt.subplots(nrows=3, ncols=len(months), figsize=(38, 18),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(nrows=2, ncols=len(months), figsize=(38, 12),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

    for ax in axes.flat:
        ax.axes.axis('tight')
        ax.set_xlabel('')

    for i, m in enumerate(months):
        t_month = select_time(temp, period[0], period[1]).groupby('time.month')[m].mean('time')
        w_month = select_time(wetbulb, period[0], period[1]).groupby('time.month')[m].mean('time')
        r_month = t_month - w_month

        axes[0, i].coastlines(zorder=2)  # zorder=2 > zorder=1 so the coastlines will be plotted over the ocean mask
        axes[0, i].set_extent(extent, crs=projection)  # set lat,lon borders of plot
        axes[0, i].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)  # mask oceans

        axes[1, i].coastlines(zorder=2)
        axes[1, i].set_extent(extent, crs=projection)
        axes[1, i].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)

        if include_residual:
            axes[2, i].coastlines(zorder=2)
            axes[2, i].set_extent(extent, crs=projection)
            axes[2, i].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)
        else:
            pass

        if reference_period is None:
            pass
        else:
            t_ref = select_time(temp, ref_period[0], ref_period[1]).groupby('time.month')[m].mean('time')
            w_ref = select_time(wetbulb, ref_period[0], ref_period[1]).groupby('time.month')[m].mean('time')
            r_ref = t_ref - w_ref

            t_month = t_month - t_ref
            w_month = w_month - w_ref
            r_month = r_month - r_ref

        # Plot colormaps from xarray
        t = t_month.plot.pcolormesh(
            ax=axes[0, i], vmin=temp_min, vmax=temp_max, cmap='bwr',
            add_colorbar=False, extend='both')

        w = w_month.plot.pcolormesh(
            ax=axes[1, i], vmin=temp_min, vmax=temp_max, cmap='bwr',
            add_colorbar=False, extend='both')

        if include_residual:
            r = r_month.plot.pcolormesh(
                ax=axes[2, i], vmin=temp_min, vmax=temp_max, cmap='bwr',
                add_colorbar=False, extend='both')

            grid_r = axes[2, i].gridlines(crs=projection, draw_labels=True)
            grid_r.top_labels = False
            grid_r.right_labels = False

        else:
            pass

        # Add titles to top of columns
        axes[0, i].set_title(calendar.month_name[m])

        # Add grid and gridlabels to the bottom and left side of axes
        grid_t = axes[0, i].gridlines(crs=projection, draw_labels=True)
        grid_t.top_labels = False
        grid_t.right_labels = False
        grid_w = axes[1, i].gridlines(crs=projection, draw_labels=True)
        grid_w.top_labels = False
        grid_w.right_labels = False

    # Add colorbar over rightmost axis
    fig.colorbar(t, ax=axes[:, 0:len(months)], location='right', extend='both', label='degC')

    # Add ylabels since ax.set_ylabel is broken in cartopy: https://stackoverflow.com/questions/35479508/cartopy-set-xlabel-set-ylabel-not-ticklabels
    axes[0, 0].text(-0.3, 0.55, 'Drybulb temp', va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=axes[0, 0].transAxes)

    axes[1, 0].text(-0.3, 0.55, 'Wetbulb temp', va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=axes[1, 0].transAxes)

    if include_residual:
        axes[2, 0].text(-0.3, 0.55, 'Difference', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=axes[2, 0].transAxes)
    else:
        pass

    fig.suptitle('Increased temperatures in {}-{} w.r.t. reference period'.format(period[0][0:4], period[1][0:4]),
                 fontsize=16, y=1.02)
    plt.savefig('monthly_comparison-(2020-2030).png', dpi=300)


monthly_comparison_plot(mean_ensemble(temp, 'TREFHT'), mean_ensemble(wetbulb, 'wetbulb'), display_period,
                        reference_period=ref_period, include_residual=False)


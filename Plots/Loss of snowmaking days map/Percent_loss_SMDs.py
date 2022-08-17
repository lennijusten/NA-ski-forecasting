import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cpf
import cartopy.feature as cfeature
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import cmaps

df = pd.read_csv('ski_locations_w_snowmaking_days_extended.csv')

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
    # base polygon is a rectangle, another polygon is simplified switzerland
    msk = Polygon(rect_from_bound(*exts)).difference(poly[0].simplify(0.01))
    msk_pc = projection.project_geometry(msk, datacoord)  # project geometry to the projection
    return states_provinces, poly, poly2, msk_pc


states_provinces, poly, poly2, msk_pc = make_geometries()

# # create fig and axes using intended projection
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
# ax.add_geometries(poly2, crs=ccrs.PlateCarree(), facecolor='black', edgecolor='black')


# Inputs
snow_thresh = -2.0
months = [11,12,1,2,3] # if months = none, add full year
periods = [2030,2060,2090]

ref_period = ('1980-01-01 00:00:00', '2000-01-01 00:00:00')


def SMD_scatter(df, thresh, periods, months=None, projection=ccrs.LambertConformal(),
               extent=[-130, -65, 24, 50], reference_period=None):
    datacoord = ccrs.PlateCarree()

    # Setup figure
    fig, axes = plt.subplots(nrows=len(periods), ncols=1, figsize=(15, 15), subplot_kw={'projection': projection})
    fig.subplots_adjust(wspace=0, hspace=0)


    for i, p in enumerate(periods):
        # setup projection and axis properties
        axes[i].set_extent(extent, crs=datacoord)  # set lat,lon borders of plot
        axes[i].add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=1)  # mask oceans
        axes[i].add_feature(states_provinces, edgecolor='dimgray', zorder=2)
        axes[i].coastlines(zorder=4, linewidth=2)
        axes[i].add_feature(cfeature.BORDERS, zorder=5)
        axes[i].add_geometries(poly2, crs=projection, facecolor='white', edgecolor='none', zorder=3)
        axes[i].add_geometries(poly, crs=projection, facecolor='none', edgecolor='black', zorder=4)
        axes[i].add_geometries(msk_pc, projection, zorder=4, facecolor='white', edgecolor='none', alpha=1)
        axes[i].axis('off')

        # Add ylabels
        axes[i].text(0.08, 0.55, '{}'.format(p), va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=axes[i].transAxes, zorder=5)

        if reference_period is None:
            axes[0].set_title('Ensemble Mean Percent Reduction in SMDs')

        else:
            axes[0].set_title('Ensemble Mean Percent Reduction in SMDs (1980-2000 ref period)')

        smd = axes[i].scatter(x=df['Ski_Long'], y=df['Ski_Lat'], c=df['{} percent loss SMD'.format(p)],
                              cmap=cmaps.GMT_panoply,
                              vmin=0,
                              vmax=1,
                              s=50,
                              alpha=1,
                              transform=datacoord,
                              edgecolor='k',
                              linewidths=0.8,
                              zorder=7)  ## Important

    plt.savefig('SMD_map.png', dpi=300)

    fig, ax = plt.subplots()
    ax.axis('off')
    cbar = fig.colorbar(smd, location='top', label="Percent loss of snowmaking days since 2000")
    cbar.set_ticks([0.0, .2, .4, .6, .8, 1.0])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.savefig('SMD_colorbar.png', dpi=300)


SMD_scatter(df, snow_thresh, periods, months, reference_period=ref_period)
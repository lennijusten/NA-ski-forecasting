import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import glob
from natsort import natsorted
import numpy as np
from tqdm import tqdm
import joypy

wetbulb_path = '/adhara_a/ljusten/Wetbulb-bc/*.nc'
temp_path = '/adhara_a/ljusten/TREFHT-bc-degC/*.nc'

ensmems = list(range(1, 36, 1)) + list(range(101, 106, 1))


# Consider looping through each member
# Save daily climatology for each member as a datarray

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
        temp_iter = xr.open_dataset(openme_temp[i], chunks={'lat': 5, 'lon': 5})
        wetbulb_iter = xr.open_dataset(openme_wetbulb[i], chunks={'lat': 5, 'lon': 5})

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
periods = [['2020-01-01 00:00:00', '2020-12-31 00:00:00'], ['2030-01-01 00:00:00', '2030-12-31 00:00:00'], ['2040-01-01 00:00:00', '2040-12-31 00:00:00'],
           ['2050-01-01 00:00:00', '2050-12-31 00:00:00'],['2060-01-01 00:00:00', '2060-12-31 00:00:00'],['2070-01-01 00:00:00', '2070-12-31 00:00:00'],
           ['2080-01-01 00:00:00', '2080-12-31 00:00:00'],['2090-01-01 00:00:00', '2090-12-31 00:00:00'],['2100-01-01 00:00:00', '2100-12-31 00:00:00']]
months = [11,12]
p_label = ['2020','2030','2040','2050','2060','2070','2080','2090','2100']
coords = [[45,268],[39,253],[41,253],[44,289]]
coords_labels = ['Trollhaugen, WI','Aspen Snowmass, CO','Big Sky, MT','Loon Mountain, NH']
smd_thresh = -5.0

def select_coord(var, lat, lon):
    return var.sel(lat=lat,lon=lon)


decades = [int(y) for y in np.unique(wetbulb['time.year']) if y % 10 == 0]
decades_labels = [int(y) if y % 20 == 0 else '' for y in decades]


df_list = []
df_list_smd = []
for i, c in tqdm(enumerate(coords)):
    wb0 = select_coord(wetbulb, c[0], c[1])
    wb1 = wb0.sel(time=wb0.time.dt.month.isin(months))
    wb2 = wb1.sel(time=wb1.time.dt.year.isin(decades))
    wb3 = wb2.groupby('time.year')

    mean = wb3.mean('time')

    df = mean['wetbulb'].to_dataframe()
    df.reset_index(inplace=True)
    df_list.append(df)

    # -------------------------------------
    wetbulb_bool = wb1 <= smd_thresh
    wb2_smd = wetbulb_bool.groupby('time.year').sum()
    wb3_smd = wb2_smd.sel(year=wb2_smd.year.isin(decades))

    df_smd = wb3_smd['wetbulb'].to_dataframe()
    df_smd.reset_index(inplace=True)
    df_list_smd.append(df_smd)

dmin = []
dmax = []
dmean = []
for i in df_list:
    d_mean = i.groupby('year').mean()['wetbulb']
    dmean.append(d_mean)
    dmin.append(d_mean.min())
    dmax.append(d_mean.max())

norm = plt.Normalize(np.min(dmin), np.max(dmax))
original_cmap = plt.cm.autumn_r
sm = plt.cm.ScalarMappable(cmap=original_cmap, norm=norm)
sm.set_array([])

dmin_smd = []
dmax_smd = []
dmean_smd = []
for i in df_list_smd:
    d_mean_smd = i.groupby('year').mean()['wetbulb']
    dmean_smd.append(d_mean_smd)
    dmin_smd.append(d_mean_smd.min())
    dmax_smd.append(d_mean_smd.max())

norm_smd = plt.Normalize(np.min(dmin_smd), np.max(dmax_smd))
original_cmap_smd = plt.cm.cool
sm_smd = plt.cm.ScalarMappable(cmap=original_cmap_smd, norm=norm_smd)
sm_smd.set_array([])


def wetbulb_ridge(df, cmap, save_path):
    fig, axes = joypy.joyplot(
        data=df_list[i][['wetbulb', 'year']],
        by='year',
        figsize=(8, 6),
        kind="kde",
        tails=0.25,
        range_style='own',
        linewidth=1,
        grid='y',
        labels=decades_labels,
        colormap=cmap
    )

    for ax in axes:
        ax.set_xlim([-15, 10])

    axes[-1].set_xticks([-15, -10, -5, 0, 5, 10])

    # plt.xlabel("Wet-bulb temperature ({}C)".format(u'\N{DEGREE SIGN}'))
    plt.savefig(save_path, dpi=300)
    plt.show()


def snowmaking_ridge(df, cmap, save_path):
    fig, axes = joypy.joyplot(
        data=df_list[i][['wetbulb', 'year']],
        by='year',
        figsize=(8, 6),
        kind="kde",
        tails=0.25,
        range_style='own',
        linewidth=1,
        labels=decades_labels,
        grid='y',
        colormap=cmap
    )

    for ax in axes:
        ax.set_xlim([0, 70])

    axes[-1].set_xticks([0, 10, 20, 30, 40, 50, 60, 70])

    # plt.xlabel('Snowmaking days')
    plt.savefig(save_path, dpi=300)
    plt.show()


save_path = ['wetbulb-ridgeline-trollhaugen.png', 'wetbulb-ridgeline-aspen.png', 'wetbulb-ridgeline-big_sky.png',
             'wetbulb-ridgeline-loon_mountain.png']
save_path_smd = ['smd-ridgeline-trollhaugen.png', 'smd-ridgeline-aspen.png', 'smd-ridgeline-big_sky.png',
                 'smd-ridgeline-loon_mountain.png']

for i in range(len(coords)):
    cmap = colors.ListedColormap(original_cmap(norm(dmean[i])))
    axes = wetbulb_ridge(df_list[i], cmap, save_path[i])

fig, ax = plt.subplots()
ax.axis('off')
cbar = fig.colorbar(sm, label='Ensemble mean ({}C)'.format(u'\N{DEGREE SIGN}'), location='bottom',
                    ticks=[-10, -8, -6, -4, -2, 0, 2])
cbar.ax.set_xticklabels([-10, -8, -6, -4, -2, 0, 2])
plt.savefig('ridgeline_colorbar.png', dpi=300)
plt.show()

for i in range(len(coords)):
    cmap = colors.ListedColormap(original_cmap_smd(norm_smd(dmean_smd[i])))
    snowmaking_ridge(df_list_smd[i], cmap, save_path_smd[i])

fig, ax = plt.subplots()
ax.axis('off')
cbar = fig.colorbar(sm_smd, label='Ensemble mean (snowmaking days)', location='bottom')
plt.savefig('ridgeline_smd_colorbar.png', dpi=300)
plt.show()

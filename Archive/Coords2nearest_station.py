import pandas as pd
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # This can cause security concerns for unverified websites

ski_resorts = pd.read_csv('/Users/Lenni/Downloads/ski_coords.csv')

df = pd.read_csv('ftp://ftp.ncei.noaa.gov/pub/data/noaa/isd-history.csv')
df['year1'] = df['BEGIN'].astype(str).str[:4]
df['year2'] = df['END'].astype(str).str[:4]


def distance_from_coords(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # units km


def get_sensors_in_range(radius, min_years, return_nearest=True):
    # return_nearest returns the nearest sensor if there are no sensors within the radius
    df['n_years'] = df['year2'].astype(int) - df['year1'].astype(int)
    df_inrange = df.loc[(df['dist'] <= radius) & (df['n_years'] >= min_years)]

    if return_nearest and len(df_inrange == 0):
        df_inrange = df.iloc[0]

    return df_inrange


# df['dist'] = np.hypot(df['LAT']-lat, df['LON']-lon)


# Just unique ski resort optimal in-range sensor info.
# If multiple sensors are in-range then the sensor with largest elevation is selected
candidates_resorts = pd.DataFrame()
candidates = pd.DataFrame()  # All sensors with ski resorts info
for r in range(len(ski_resorts)):
    # option1: ski resort has no sensors within radius
    #   - drop ski resort from candidate list
    # option2: ski resort has one sensor within radius
    #   - In this case we want to just append the sensor/ski-resort pair
    # option3: ski resorts has multiple sensors within radius
    #   - In this case we want to compare for the closest elevation/distance trade-off

    lat_r = ski_resorts['lat'].iloc[r]
    lon_r = ski_resorts['lon'].iloc[r]

    df['dist'] = distance_from_coords([lat_r] * len(df), [lon_r] * len(df), df['LAT'].values, df['LON'].values)
    df = df.sort_values(by='dist', ascending=True)

    df_inrange = get_sensors_in_range(10, 30, return_nearest=False)
    resort = ski_resorts.iloc[r]
    df_inrange2 = df_inrange.assign(**resort)  # copy ski resort information to each row if more than one row
    if len(df_inrange) == 0:
        continue
    elif len(df_inrange) == 1:
        candidates = candidates.append(df_inrange2)
        highest_sensor = df_inrange.sort_values(by='ELEV(M)', ascending=False).iloc[0]
        candidates_resorts = candidates_resorts.append(pd.concat([resort, highest_sensor], axis=0), ignore_index=True)
    else:
        candidates = candidates.append(df_inrange2)  # for now we allow multiple entries per resort
        highest_sensor = df_inrange.sort_values(by='ELEV(M)', ascending=False).iloc[0]
        candidates_resorts = candidates_resorts.append(pd.concat([resort, highest_sensor], axis=0), ignore_index=True)


candidates_resorts = candidates_resorts[['name', 'dist', 'lat', 'lon', 'STATION NAME', 'LAT', 'LON', 'ELEV(M)',
                                         'CTRY', 'STATE', 'year1', 'year2', 'BEGIN', 'END', 'ICAO', 'USAF', 'WBAN',
                                         'Page_URL']]

# year1=2010
# year2=2013
# usaf = '726410'  #Grab this number from the table above
# wban = '14837'   #Grab this number from the table above
# usaf_all=df['USAF'] # Unique station ID
# ourstation=df['STATION NAME'][usaf_all==usaf].values[0]
# print(ourstation)

def get_isd_data(year1, year2, usaf, wban):
    obdf = []
    for yy in range(year1, year2):
        year = yy
        obdf.append(pd.read_csv(f'https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz',
                                compression='gzip', delim_whitespace=True,
                                names=['year', 'month', 'day', 'hour', 't', 'td', 'slp', 'wdir', 'wspd', 'skyc',
                                       'prcp01', 'prcp06'],
                                parse_dates={'time': ['year', 'month', 'day', 'hour']},
                                converters={
                                    't': lambda x: int(x) / 10.0,
                                    'td': lambda x: int(x) / 10.0,
                                    'slp': lambda x: int(x) / 10.0,
                                    'wspd': lambda x: int(x) / 10.0,
                                    'prcp01': lambda x: int(x) / 10.0 if int(x) >= 0 else np.nan,
                                    'prcp06': lambda x: int(x) / 10.0 if int(x) >= 0 else np.nan,
                                }))
    obdf = pd.concat(obdf, ignore_index=True)
    # obdf.head(15)
    return obdf

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pickle
import numpy as np
import datetime as dt
import pandas as pd

coord = 1  # specify coord to generate plot [0,1,2,3]
coords_labels = ['Trollhaugen, WI','Aspen Snowmass, CO','Big Sky, MT','Loon Mountain, NH']
cl_short = ['Troll','Aspen','Big-Sky','Loon']

# Get last_date data
with open('mean_last_date-coord{}.pkl'.format(coord), 'rb') as handle:
    mean_ld1 = pickle.load(handle)

with open('std_last_date-coord{}.pkl'.format(coord), 'rb') as handle:
    std_ld1 = pickle.load(handle)

mean_ld1 = pd.Series(mean_ld1)
std_ld1 = pd.Series(std_ld1).astype(float)

ld1_upper = pd.Series([mean_ld1[i] + pd.offsets.DateOffset(days=d) for i, d in enumerate(std_ld1)])
ld1_lower = pd.Series([mean_ld1[i] - pd.offsets.DateOffset(days=d) for i, d in enumerate(std_ld1)])

ld1_upper = pd.to_datetime(ld1_upper.dt.strftime('%m-%d'), format='%m-%d')
ld1_lower = pd.to_datetime(ld1_lower.dt.strftime('%m-%d'), format='%m-%d')

time = np.arange(1920, 2101, 1)
ld_data = pd.to_datetime(mean_ld1.dt.strftime('%m-%d'), format='%m-%d')
pseudo = pd.to_datetime(pd.Series(['11-15'] * len(time)), format='%m-%d') - pd.offsets.DateOffset(years=1)

# Get first_date data
with open('mean_first_date-coord{}.pkl'.format(coord), 'rb') as handle:
    mean_fd1 = pickle.load(handle)

with open('std_first_date-coord{}.pkl'.format(coord), 'rb') as handle:
    std_fd1 = pickle.load(handle)

mean_fd1 = pd.Series(mean_fd1)
std_fd1 = pd.Series(std_fd1).astype(float)

fd1_upper = pd.Series([mean_fd1[i] + pd.offsets.DateOffset(days=d) for i, d in enumerate(std_fd1)])
fd1_lower = pd.Series([mean_fd1[i] - pd.offsets.DateOffset(days=d) for i, d in enumerate(std_fd1)])

fd1_upper = pd.to_datetime(fd1_upper.dt.strftime('%m-%d'), format='%m-%d')[1:]
fd1_upper = fd1_upper - pd.offsets.DateOffset(years=1)
fd1_lower = pd.to_datetime(fd1_lower.dt.strftime('%m-%d'), format='%m-%d')[1:]
fd1_lower = fd1_lower - pd.offsets.DateOffset(years=1)

fd_data = pd.to_datetime(mean_fd1.dt.strftime('%m-%d'), format='%m-%d')[1:]
fd_data = fd_data - pd.offsets.DateOffset(years=1)


fig, ax = plt.subplots(figsize=(8, 8))
ax.fill_between(time,ld_data,fd_data,alpha=0.8)

#ax.plot(time,ld_data,c='r',linewidth=2)
#ax.plot(time,fd_data,c='r',linewidth=2)

# #488cc4
ax.fill_between(time,ld1_upper,ld1_lower,facecolor='#787878',alpha=0.6)
ax.fill_between(time,fd1_upper,fd1_lower,facecolor='#787878',alpha=0.6)

#ax.plot(time, data)
#ax.plot(time,pseudo,c='C0')

date_form = DateFormatter("%m-%d")
ax.yaxis.set_major_formatter(date_form)
ax.set_xlim([time[0],time[-1]])
ax.set_ylim([dt.date(1899, 10, 1), dt.date(1900, 6, 1)])
# ax.set_ylabel('Start date                                                                                  End date')
plt.title(coords_labels[coord])
plt.savefig('season-length-{}.png'.format(cl_short[coord]),dpi=300)
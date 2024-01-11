#%%
from geo.io import read_csv_to_geodataframe
from geo.trajectory_cleaner import traj_clean_drift

traj = read_csv_to_geodataframe('../data/cells/004.csv')
traj.loc[:, 'rid'] = 1 
_records = traj_clean_drift(traj, ['rid', 'dt', 'geometry'], dislimit=5000)
_records.shape

#%%
set(_records.index).difference(set(traj.index))

# %%

import pandas as pd
from os import path
import xarray as xr
import numpy as np

from gdc.utils import GDC_DATA_PATH

# 30' simulated temperature data - unknown dates
df_temp_simulated = pd.read_feather(
        path.join(GDC_DATA_PATH, 'simulated', 'temperature_export.feather')
)

# 1h real temperature data - Jan 1 to Dec 31, 2023
ds_temp_real = xr.open_dataset(
    path.join(GDC_DATA_PATH, 'real', 'temperature',
              '91027002_ORLY_MTO_1H_2023.nc'),
    engine='netcdf4'
)

df_temp_real = ds_temp_real[['time', 'ta']].to_dataframe().reset_index()


# finding offset

centered_df_temp_simulated = df_temp_simulated.mean(axis=0)
centered_df_temp_simulated -= centered_df_temp_simulated.mean()
#get even indices only
centered_df_temp_simulated = centered_df_temp_simulated.T.iloc[::2].reset_index(
    drop=True).T

centered_df_temp_real = df_temp_real.copy()
centered_df_temp_real['ta'] -= centered_df_temp_real['ta'].mean()


def roll(s, k):
    # works for both series and dataframes
    return s.iloc[np.roll(np.arange(len(s)), k)]


def err_fit(k):
    r = roll(centered_df_temp_simulated, k).reset_index(drop=True)
    return np.mean(np.abs(r - centered_df_temp_real['ta']))


ks = list(range(-2500, -1500))
errs = [err_fit(k) for k in ks]
offset_idx = int(np.argmin(errs))
offset = ks[offset_idx]

# Apply offset to simulated data & create datetime index for both dataframes
df_temp_simulated_normalized = roll(
    df_temp_simulated.T.iloc[::2].reset_index(drop=True),
    offset).reset_index(drop=True).T



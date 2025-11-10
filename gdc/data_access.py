import pandas as pd
from os import path
import xarray as xr
import numpy as np

from gdc.utils import GDC_DATA_PATH

__all__ = [
    'df_temp_simulated_normalized', 'df_temp_real', 'demean',
    'df_load_simulated_normalized', 'df_hourly_load_real', 'df_labels',
    'to_zero_one', 'df_hourly_prices', 'CB', 'df_merged_real'
]

#########################################
# Step 1 Load Temperatures / find offset
#########################################

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

def demean(df):
    centered_df = df.copy()
    if isinstance(centered_df, pd.DataFrame):
        for col in centered_df.columns:
            centered_df[col] -= centered_df[col].mean()
        return centered_df
    else:
        return centered_df - centered_df.mean()


centered_df_temp_simulated = demean(df_temp_simulated.mean(axis=0))

# get even indices only
centered_df_temp_simulated = (
    centered_df_temp_simulated.T.iloc[::2].reset_index(drop=True).T)

centered_df_temp_real = demean(df_temp_real['ta'])


def roll(s, k):
    # works for both series and dataframes
    return s.iloc[np.roll(np.arange(len(s)), k)]


def err_fit(k):
    r = roll(centered_df_temp_simulated, k).reset_index(drop=True)
    return np.mean(np.abs(r - centered_df_temp_real))


ks = list(range(-2500, -1500))
errs = [err_fit(k) for k in ks]
offset_idx = int(np.argmin(errs))
offset = ks[offset_idx]

# Apply offset to simulated data & create datetime index for both dataframes
df_temp_simulated_normalized = roll(
    df_temp_simulated.T.iloc[::2].reset_index(drop=True),
    offset).reset_index(drop=True).T

df_temp_simulated_normalized = df_temp_simulated_normalized.T
df_temp_simulated_normalized.index = pd.to_datetime(
    df_temp_real.iloc[:len(df_temp_simulated_normalized)]['time'].values)

df_temp_real.set_index('time', inplace=True)


#########################################
# Step 2 Apply Offset to simulated loads
#########################################

df_load_simulated = pd.read_feather(
    path.join(GDC_DATA_PATH, 'simulated', 'load_curve_export.feather')
)
df_load_simulated = df_load_simulated.T
df_load_simulated = df_load_simulated.iloc[::2].reset_index(drop=True)
df_load_simulated_normalized = roll(df_load_simulated, offset).reset_index(drop=True)
df_load_simulated_normalized.index = pd.to_datetime(
    df_temp_real.iloc[:len(df_load_simulated)].index.values)


df_load_real = pd.read_csv(
    path.join(GDC_DATA_PATH, 'real', 'demande_rte_2023.csv')
)

df_load_real = df_load_real.drop(columns=['Périmètre', 'Nature'])

# combine Date and Heures into a proper datetime column
df_load_real['datetime'] = pd.to_datetime(
    df_load_real['Date'] + ' ' + df_load_real['Heures'])

# round to the nearest hour (or floor if you prefer)
df_load_real['datetime'] = df_load_real['datetime'].dt.floor('h')

# group by hour and sum the numeric columns
df_hourly_load_real = df_load_real.groupby('datetime')[
    ['Consommation', 'Prévision J-1', 'Prévision J']
].sum()


def to_zero_one(df):
    return (df - df.min()) / (df.max() - df.min())


################################################
# Step 3 Load simulation labels and real prices
################################################

df_labels = pd.read_feather(
    path.join(GDC_DATA_PATH, 'simulated', 'labels_export.feather')
)

df_hourly_prices = pd.read_csv(
    path.join(GDC_DATA_PATH, 'real', 'prices_fr.csv')
)

df_hourly_prices['date'] = pd.to_datetime(df_hourly_prices['Datetime (UTC)'])
df_hourly_prices = df_hourly_prices.set_index('date')
df_hourly_prices = df_hourly_prices.loc['2023', ['Price (EUR/MWhe)']]


df_merged_real = df_temp_real[['ta']].join(
    df_hourly_load_real['Consommation'], how='inner').join(
    df_hourly_prices, how='inner')


class CodeBook:
    cons = 'Consommation'
    price = 'Price (EUR/MWhe)'
    day_ahead_predicted_cons = 'Prévision J-1'
    day_of_predicted_cons = 'Prévision J'
    temp = 'ta'


CB = CodeBook

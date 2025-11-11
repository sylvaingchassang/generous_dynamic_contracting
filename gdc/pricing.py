import pandas as pd
from os import path
from datetime import datetime, date

from gdc.data_access import GDC_DATA_PATH


def _load_base_tariff_df():
    df = pd.read_csv(
        path.join(GDC_DATA_PATH, 'real', 'retail_pricing', 'Option_Base.csv'),
        sep=";",
        decimal=",",
        dtype={"P_SOUSCRITE": "float"},
        parse_dates=["DATE_DEBUT", "DATE_FIN"],
        dayfirst=True,
        encoding="utf-8"
    )
    return df

df_base_tarif = _load_base_tariff_df()
_BASE_TARIFF_DF = df_base_tarif

def _ensure_date(x):
    if isinstance(x, pd.Timestamp):
        return x.normalize()
    if isinstance(x, date) and not isinstance(x, datetime):
        return pd.Timestamp(x)
    if isinstance(x, datetime):
        return pd.Timestamp(x.date())
    if isinstance(x, str):
        try:
            return pd.to_datetime(x, dayfirst=True).normalize()
        except Exception:
            return pd.to_datetime(x).normalize()


def get_base_price(day, subscribed_power, kwh, include_breakdown=False):
    ts = _ensure_date(day)
    sp = float(subscribed_power)

    match = _BASE_TARIFF_DF.loc[
        (_BASE_TARIFF_DF["DATE_DEBUT"] <= ts)
        & (_BASE_TARIFF_DF["DATE_FIN"] >= ts)
        & (_BASE_TARIFF_DF["P_SOUSCRITE"] == sp)
    ].sort_values("DATE_DEBUT", ascending=False).iloc[0]

    days_year = 365

    # TTC
    fixed_daily_ttc = float(match["PART_FIXE_TTC"]) / days_year
    unit_var_ttc = float(match["PART_VARIABLE_TTC"])
    variable_ttc = unit_var_ttc * float(kwh)
    total_ttc = fixed_daily_ttc + variable_ttc

    # HT
    fixed_daily_ht = float(match["PART_FIXE_HT"]) / days_year
    unit_var_ht = float(match["PART_VARIABLE_HT"])
    variable_ht = unit_var_ht * float(kwh)
    total_ht = fixed_daily_ht + variable_ht

    breakdown = {
        "fixed_daily_ttc": fixed_daily_ttc,
        "variable_ttc": variable_ttc,
        "unit_variable_ttc": unit_var_ttc,
        "total_ttc": total_ttc,
        "fixed_daily_ht": fixed_daily_ht,
        "variable_ht": variable_ht,
        "unit_variable_ht": unit_var_ht,
        "total_ht": total_ht
    }

    if include_breakdown:
        return total_ttc, total_ht, breakdown
    return total_ttc, total_ht






def compute_simulated_base_variable_profit_ht(
        subscribed_power_kva=6.0, loads_df=None, prices_df=None):
    """
    Compute simulated firm profit for base retail contracts.

    Profit is defined hourly as:
        [base retail unit price (HT) - spot price] * load.

    Parameters
    ----------
    subscribed_power_kva : float
        Subscribed power in kVA to use for all consumers (e.g., 3.0, 6.0, 9.0).
    loads_df : pandas.DataFrame or None
        Optional loads dataframe (hours x consumers). If None, uses
        gdc.data_access.df_load_simulated_normalized.
    prices_df : pandas.DataFrame or None
        Optional prices dataframe with a column 'Price_EUR_MWh' indexed by hour.
        If None, uses gdc.data_access.df_hourly_prices.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by hourly timestamps, columns are consumer IDs, values
        are profits in EUR (HT) for each hour.
    """

    # Ensure DatetimeIndex and optionally time-slice
    loads = loads_df.copy()
    prices = prices_df.copy()

    # Align to common hourly index
    idx = loads.index.intersection(prices.index)
    loads = loads.loc[idx]
    prices = prices.loc[idx]

    # Build daily series of unit variable HT price for the subscribed power
    # Compute once per unique day, then expand to hours
    days = pd.Index(pd.to_datetime(idx.date)).unique()

    unit_ht_by_day = {}
    for d in days:
        _, _, br = get_base_price(
            d, subscribed_power_kva, kwh=0.0, include_breakdown=True)
        unit_ht_by_day[pd.Timestamp(d).normalize()] = float(
            br['unit_variable_ht'])

    daily_ht_variable_series = pd.Series(unit_ht_by_day).sort_index()

    # Map each hour to its day's unit price
    day_index = pd.to_datetime(idx.date)
    unit_ht_hourly = pd.Series(day_index).map(daily_ht_variable_series).values

    # Convert spot EUR/MWh to EUR/kWh
    if 'Price_EUR_MWh' not in prices.columns:
        raise ValueError("prices_df must contain column 'Price_EUR_MWh'")
    spot_per_kwh = prices['Price_EUR_MWh'].astype(float).values / 1000.0

    # Margin per kWh, then multiply by loads (assumed kWh per hour per consumer)
    margin_per_kwh = unit_ht_hourly - spot_per_kwh

    # Multiply each row of loads by the corresponding scalar margin for
    # that hour
    profit_df = loads.mul(margin_per_kwh, axis=0)
    
    return profit_df


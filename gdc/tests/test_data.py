import pytest
import pandas as pd
import xarray as xr
import numpy as np
import unittest

from gdc.data_access import (df_temp_simulated, df_temp_real, 
                             df_temp_simulated_normalized)
from gdc.tests.testutils import CachedTestCase


def test_df_temp_simulated_exists():
    """Test that the simulated temperature dataframe exists."""
    assert df_temp_simulated is not None
    assert isinstance(df_temp_simulated, pd.DataFrame)


def test_df_temp_simulated_structure():
    """Test the structure of the simulated temperature dataframe."""
    # Check that the dataframe is not empty
    assert not df_temp_simulated.empty

    # The dataframe appears to have numeric column names
    # Check that the columns are of the expected type
    assert all(col.isdigit() for col in df_temp_simulated.columns)

    # Check that there are multiple columns (time points)
    assert len(df_temp_simulated.columns) > 0


def test_df_temp_real_exists():
    """Test that the real temperature dataframe exists."""
    assert df_temp_real is not None
    assert isinstance(df_temp_real, pd.DataFrame)


def test_df_temp_real_structure():
    """Test the structure of the real temperature dataframe."""
    # Check that the dataframe is not empty
    assert not df_temp_real.empty

    # Check that the dataframe has the expected columns
    assert 'time' in df_temp_real.columns
    assert 'ta' in df_temp_real.columns

    # Check that the time column is of datetime type
    assert pd.api.types.is_datetime64_any_dtype(df_temp_real['time'])

    # Check that the temperature column has numeric values
    assert pd.api.types.is_numeric_dtype(df_temp_real['ta'])


def test_df_temp_real_values():
    """Test some basic properties of the real temperature values."""
    # Temperature appears to be in Kelvin
    # Reasonable range for atmospheric temperatures in Kelvin
    assert df_temp_real['ta'].min() > 200
    assert df_temp_real['ta'].max() < 330

    # Check for missing values
    assert not df_temp_real['ta'].isna().any()


def test_df_temp_simulated_values():
    """Test some basic properties of the simulated temperature values."""
    # Check that all values in the dataframe are numeric
    for col in df_temp_simulated.columns:
        assert pd.api.types.is_numeric_dtype(df_temp_simulated[col])

    # Check that the values are within a reasonable range for temperatures
    # Assuming these are in Celsius based on the values observed
    min_temp = df_temp_simulated.values.min()
    max_temp = df_temp_simulated.values.max()

    assert min_temp > -50
    assert max_temp < 60

    # Check for missing values
    assert not df_temp_simulated.isna().any().any()


def test_df_temp_simulated_shape():
    """Test the shape of the simulated temperature dataframe."""
    # Check that the dataframe has the expected shape
    shape = df_temp_simulated.shape

    # We expect a 2D dataframe with rows and columns
    assert len(shape) == 2
    assert shape[0] == 10000  # At least one row
    assert shape[1] == 17424  # At least one column


def test_df_temp_real_shape():
    """Test the shape of the real temperature dataframe."""
    # Check that the dataframe has the expected shape
    shape = df_temp_real.shape

    # We expect a 2D dataframe with rows and columns
    assert len(shape) == 2
    assert shape[0] == 8760  # At least one row
    assert shape[1] == 6  # At least one column


def test_df_temp_simulated_normalized_shape():
    assert df_temp_simulated_normalized.shape == (10000, 8712)


class TestDataframeStatistics(CachedTestCase):

    def test_df_temp_simulated_statistics(self):
        # Calculate mean and standard deviation across all values
        all_values = df_temp_simulated.values.flatten()
        mean_value = np.mean(all_values)
        std_value = np.std(all_values)

        # Create a dictionary with the statistics
        stats = {
            'mean': float(mean_value),
            'std': float(std_value)
        }

        # Compare with cached values or save new values
        # Set regenerate=True to update the cached values when needed
        self.assertEqualToCached(
            stats, 'simulated_temp_stats')
                

    def test_df_temp_real_statistics(self):
        # Calculate mean and standard deviation of temperature values
        mean_value = df_temp_real['ta'].mean()
        std_value = df_temp_real['ta'].std()

        # Create a dictionary with the statistics
        stats = {
            'mean': float(mean_value),
            'std': float(std_value)
        }

        # Compare with cached values or save new values
        # Set regenerate=True to update the cached values when needed
        self.assertEqualToCached(
            stats, 'real_temp_stats')


def test_offset():
    from gdc.data_access import offset
    assert offset == -2160
    
    
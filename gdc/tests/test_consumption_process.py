import numpy as np
import json
import pandas as pd
from numpy.testing import assert_array_almost_equal
from gdc.tests.testutils import CachedTestCase
from gdc.estimation.consumption import (
    PooledMDHUncorrelatedErrors, IndividualMDHUncorrelatedErrors)
from gdc.data_access import (
    df_load_simulated_normalized, df_temp_simulated_normalized)


class TestPooledSeasonalUncorrelatedErrors(CachedTestCase):

    def setUp(self):
        self.model = PooledMDHUncorrelatedErrors(
            y=df_load_simulated_normalized[range(100)],
            temp=df_temp_simulated_normalized[range(100)]
        )

    def test_demeaned_variables(self):
        (y_within, hdd_within, cdd_within), means = (
            self.model.demeaned_variables())
        assert y_within.shape == (8712, 100)
        assert hdd_within.shape == (8712, 100)
        assert cdd_within.shape == (8712, 100)
        y_m, y_d, y_h = means.y_means
        assert y_m.shape == (12,)
        assert y_d.shape == (7,)
        assert y_h.shape == (24,)
        self.assertAlmostEqual(y_d.mean(), 0.0, places=2)
        self.assertAlmostEqual(y_h.mean(), 0.0, places=2)

    def test_estimation_results(self):
        beta, means = self.model.fit()
        assert beta.shape == (2,)
        summary = self.model.summary(
            beta, means,
            "PooledMDHUncorrelatedErrors")
        self.assertAlmostEqualToCached(
            {'Summary': summary},
            'pooled_mdh_uncorrelated_errors_summary',
            delta=1e-6
        )

    def test_predict_static_means_out_of_sample(self):
        beta, means = self.model.fit()
        np.random.seed(42)

        original_temp_data = df_temp_simulated_normalized
        original_index = original_temp_data.index

        selected_row_indices = np.random.choice(
            len(original_index), size=100, replace=False)
        out_of_sample_datetime_index = original_index[
            selected_row_indices].sort_values()


        hdd_values = self.model.HDDv[selected_row_indices, :20]  # shape: (
        # 100, 20)
        cdd_values = self.model.CDDv[selected_row_indices, :20]  # shape: (
        # 100, 20)

        # Create DataFrames with datetime index and consumer columns
        n_consumers = hdd_values.shape[1]
        consumer_columns = [f'consumer_{i}' for i in range(n_consumers)]

        hdd_out_of_sample = pd.DataFrame(
            hdd_values,
            index=out_of_sample_datetime_index,
            columns=consumer_columns
        )
        cdd_out_of_sample = pd.DataFrame(
            cdd_values,
            index=out_of_sample_datetime_index,
            columns=consumer_columns
        )

        # Get predictions using the out-of-sample data
        predictions = self.model.predict_static_means(
            beta, means,
            hddv=hdd_out_of_sample,
            cddv=cdd_out_of_sample
        )

        self.assertAlmostEqualToCached(
            predictions.to_json(),
            'pooled_predict_static_means_out_of_sample',
            delta=1e-8
        )


class TestIndividualSeasonalUncorrelatedErrors(CachedTestCase):
    # Note: original nb model did not use individual month effects
    def setUp(self):
        self.model = IndividualMDHUncorrelatedErrors(
            y=df_load_simulated_normalized[range(100)],
            temp=df_temp_simulated_normalized[range(100)]
        )

    def test_demeaned_variables(self):
        (y_within, hdd_within, cdd_within), means = (
            self.model.demeaned_variables())
        assert y_within.shape == (8712, 100)
        assert hdd_within.shape == (8712, 100)
        assert cdd_within.shape == (8712, 100)
        y_m, y_d, y_h = means.y_means
        assert y_m.shape == (12, 100)
        assert y_d.shape == (7, 100)
        assert y_h.shape == (24, 100)
        assert_array_almost_equal(y_d.mean(axis=0), np.zeros(100))
        assert_array_almost_equal(y_h.mean(axis=0), np.zeros(100))

    def test_estimation_results(self):
        beta, means = self.model.fit()
        assert beta.shape == (2,)
        summary = self.model.summary(
            beta, means,
            "IndividualMDHUncorrelatedErrors")
        self.assertAlmostEqualToCached(
            {'Summary': summary},
            'individual_mdh_uncorrelated_errors_summary',
            delta=1e-6
        )


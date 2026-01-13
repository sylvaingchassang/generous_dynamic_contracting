import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from gdc.tests.testutils import CachedTestCase
from gdc.tempo.estimation.consumption import (
    PooledMDHUncorrelatedErrors, IndividualMDHUncorrelatedErrors, ARErrorModel)
from gdc.tempo.data_access import (
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
            label="PooledMDHUncorrelatedErrors")
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
            label="IndividualMDHUncorrelatedErrors")
        self.assertAlmostEqualToCached(
            {'Summary': summary},
            'individual_mdh_uncorrelated_errors_summary',
            delta=1e-6
        )

    def test_pool_ar1_dynamic_resids(self):
        beta, means = self.model.fit()
        error_ar1 = ARErrorModel(
            self.model.static_resids(beta, means), lags=(1,))
        error_ar1_res = error_ar1.fit()
        summary = self.model.summary(
            beta, means, error_model_=error_ar1_res,
            label="IndividualMDH with AR(1) errors")
        self.assertAlmostEqualToCached(
            {'Summary': summary},
            'individual_mdh_ar1_errors_summary', delta=1e-6)

    def test_pool_ar1_24_dynamic_resids(self):
        beta, means = self.model.fit()
        error_ar1_24 = ARErrorModel(
            self.model.static_resids(beta, means), lags=(1, 24))
        error_ar1_24_res = error_ar1_24.fit()
        summary = self.model.summary(
            beta, means, error_model_=error_ar1_24_res,
            label="IndividualMDH with AR(1, 24) errors")
        self.assertAlmostEqualToCached(
            {'Summary': summary},
            'individual_mdh_ar1_24_errors_summary', delta=1e-6)


def simulate_ar3_resids(T, N, coeffs, sigma=1.0, seed=78):
    np.random.seed(seed)
    resids = np.zeros((T, N))
    for n in range(N):
        eps = np.random.normal(0, sigma, T)
        for t in range(3, T):
            resids[t, n] = (coeffs[0] * resids[t-1, n] +
                            coeffs[1] * resids[t-2, n] +
                            coeffs[2] * resids[t-3, n] +
                            eps[t])
    return resids

def test_arerror_model_ar3_stats():
    T, N = 500, 10
    true_coeffs = [26/24, -9/24, 1/24]
    true_sigma2 = 1.0
    resids = simulate_ar3_resids(T, N, true_coeffs, sigma=true_sigma2)
    model = ARErrorModel(resids, lags=(1,2,3))
    fit_result = model.fit()
    est_coeffs = fit_result.coeffs
    est_sigma2 = fit_result.sigma2
    sample_innov = fit_result.sample_innov

    # Coefficients
    assert np.allclose(est_coeffs, true_coeffs, atol=0.05)
    # Stationarity
    assert model._check_stationarity(est_coeffs)
    # sigma2
    assert np.isclose(est_sigma2, true_sigma2, atol=0.05)
    # Sample distribution stats
    assert np.isclose(sample_innov.expectation(), 0, atol=0.05)
    assert np.isclose(sample_innov.std(), np.sqrt(true_sigma2), atol=0.1)

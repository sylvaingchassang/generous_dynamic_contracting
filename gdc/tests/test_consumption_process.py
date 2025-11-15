from gdc.tests.testutils import CachedTestCase
from gdc.estimation.consumption import (
    PooledSeasonalUncorrelatedErrors, IndividualSeasonalUncorrelatedErrors)
from gdc.data_access import (
    df_load_simulated_normalized, df_temp_simulated_normalized)


class TestPooledSeasonalUncorrelatedErrors(CachedTestCase):

    def setUp(self):
        self.model = PooledSeasonalUncorrelatedErrors(
            Y=df_load_simulated_normalized[range(100)],
            T=df_temp_simulated_normalized[range(100)]
        )

    def test_estimation_results(self):
        alpha_m, beta_A, seasonal = self.model.fit()
        assert alpha_m.shape == (12,)
        assert beta_A.shape == (2,)
        assert seasonal['Hy0'].shape == (24, 1)
        assert seasonal['Dy0'].shape == (7, 1)
        summary = self.model.print_coeffs_and_forecast_metrics(
            beta=beta_A,
            alpha_m=alpha_m,
            Hy0=seasonal['Hy0'],
            Dy0=seasonal['Dy0'],
            label="PooledSeasonalUncorrelatedErrors"
        )
        self.assertEqualToCached(
            {'Summary': summary},
            'pooled_seasonal_uncorrelated_errors_summary')


class TestIndividualSeasonalUncorrelatedErrors(CachedTestCase):

    def setUp(self):
        self.model = IndividualSeasonalUncorrelatedErrors(
            Y=df_load_simulated_normalized[range(100)],
            T=df_temp_simulated_normalized[range(100)]
        )

    def test_estimation_results(self):
        alpha_m, beta, seasonal = self.model.fit()
        assert alpha_m.shape == (12,)
        assert beta.shape == (2,)
        assert seasonal['Hy0'].shape == (24, 100)
        assert seasonal['Dy0'].shape == (7, 100)
        summary = self.model.print_coeffs_and_forecast_metrics(
            beta=beta,
            alpha_m=alpha_m,
            Hy0=seasonal['Hy0'],
            Dy0=seasonal['Dy0'],
            label="IndividualSeasonalUncorrelatedErrors",
        )
        self.assertEqualToCached(
            {'Summary': summary},
            'individual_seasonal_uncorrelated_errors_summary')
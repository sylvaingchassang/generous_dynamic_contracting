import numpy as np
import pandas as pd
from gdc.pricing import get_base_price, compute_simulated_base_profit_ht
from gdc.tests.testutils import CachedTestCase
from gdc.data_access import (df_load_simulated_normalized, df_hourly_prices,
                             get_subscribers)



class TestGetBasePrice(CachedTestCase):

    def test_get_base_price_values_cached(self):
        # Build a compact yet representative matrix of scenarios across the year
        # and common subscribed power levels.
        days = [
            '2023-01-15',
            '2023-04-15',
            '2023-07-15',
            '2023-10-15',
        ]
        subscribed_powers = [3.0, 6.0, 9.0]
        consumptions_kwh = [0.0, 10.0, 20.0]

        results = {}
        for day in days:
            for sp in subscribed_powers:
                for kwh in consumptions_kwh:
                    total_ttc, total_ht, br = get_base_price(
                        day, sp, kwh, include_breakdown=True)
                    key = f"{pd.to_datetime(day).date()}__{int(sp)}kVA__{int(kwh)}kWh"
                    results[key] = {
                        'total_ttc': float(total_ttc),
                        'total_ht': float(total_ht),
                        'fixed_daily_ttc': float(br['fixed_daily_ttc']),
                        'fixed_daily_ht': float(br['fixed_daily_ht']),
                        'unit_variable_ttc': float(br['unit_variable_ttc']),
                        'unit_variable_ht': float(br['unit_variable_ht']),
                    }

        # Auto-regenerate once if missing, then lock expectations
        ref = 'get_base_price_reference'
        self.assertEqualToCached(results, ref)

    def test_get_base_price_breakdown_consistency(self):
        # A few spot checks to ensure arithmetic consistency of the breakdown
        checks = [
            ('2023-02-01', 6.0, 0.0),
            ('2023-06-01', 6.0, 10.0),
            ('2023-11-01', 9.0, 15.5),
        ]
        for day, sp, kwh in checks:
            total_ttc, total_ht, br = get_base_price(
                day, sp, kwh, include_breakdown=True)
            recomputed_ttc = float(br['fixed_daily_ttc']) + float(
                br['unit_variable_ttc']) * float(kwh)
            recomputed_ht = float(br['fixed_daily_ht']) + float(
                br['unit_variable_ht']) * float(kwh)
            assert np.isclose(total_ttc, recomputed_ttc, rtol=0, atol=1e-9)
            assert np.isclose(total_ht, recomputed_ht, rtol=0, atol=1e-9)


class TestSimulatedProfits(CachedTestCase):

    def test_simulated_base_profits(self):
        p = 6
        this_loads = df_load_simulated_normalized.loc[:, get_subscribers(p)]
        this_profits = compute_simulated_base_profit_ht(
            p, this_loads, df_hourly_prices)

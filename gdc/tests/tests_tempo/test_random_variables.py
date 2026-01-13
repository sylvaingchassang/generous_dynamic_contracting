from gdc.tests.testutils import CachedTestCase
from gdc.tempo.estimation.random_variables import RandomVariable
import numpy as np


class TestRandomVariable(CachedTestCase):
    def test_uniform(self):
        rv = RandomVariable.uniform(0, 1)
        assert rv.expectation() == 0.4948505050505051
        assert rv.prob_gt(0.5) == 0.5
        assert rv.prob_lt(0.5) == 0.5
        self.assertEqualToCached(
            rv.sample((3, 3), 0).tolist(), 'uniform_sample')

    def test_normal(self):
        rv = RandomVariable.normal(1, 1)
        assert rv.expectation() == 1.003028494106667
        assert rv.prob_gt(1) == 0.5
        assert rv.prob_lt(1) == 0.5
        self.assertEqualToCached(
            rv.sample((3, 3), 0).tolist(), 'normal_sample')

    def test_lognormal(self):
        rv = RandomVariable.lognormal(1, 1, 1000)
        log_rv = rv.apply(np.log)
        assert rv.expectation() == 4.430789859873528
        assert log_rv.expectation() == 1.001087138980536
        assert log_rv.std() == 0.9886764592996713
        assert rv.prob_gt(1) == 0.842
        assert rv.prob_lt(1) == 0.158
        self.assertEqualToCached(
            rv.sample((3, 3), 0).tolist(), 'lognormal_sample')

    def test_gumbel(self):
        rv = RandomVariable.gumbel(1, 1)
        assert rv.expectation() == 1.5492427331783105
        assert rv.prob_gt(1) == 0.63
        assert rv.prob_lt(1) == 0.37
        self.assertEqualToCached(
            rv.sample((3, 3), 0).tolist(), 'gumbel_sample')

    def test_add(self):
        rv1 = RandomVariable.uniform(0, 1)
        rv2 = RandomVariable.uniform(0, 1)
        rv = rv1 + rv2
        assert rv.expectation() == 0.9884271400877461
        assert rv.prob_gt(2) == 0
        assert rv.prob_lt(0) == 0
        assert rv.prob_lt(1) == .5
        self.assertEqualToCached(
            rv.values.tolist(), 'sum_uniform_distribution')

    def test_add_constant(self):
        rv = RandomVariable.uniform(0, 1)
        rv = rv + 1
        assert rv.expectation() == 1.474850505050505
        assert rv.prob_gt(2) == 0
        assert rv.prob_lt(1) == 0
        assert rv.prob_lt(1.5) == .5
        self.assertEqualToCached(
            rv.values.tolist(), 'uniform_plus_1')

    def test_minus(self):
        rv = - 2 * RandomVariable.uniform(0, 1)
        assert rv.expectation() == -.9897010101010102
        assert rv.prob_gt(0) == 0
        assert rv.prob_lt(-2) == 0
        assert rv.prob_lt(-1) == .5
        self.assertEqualToCached(
            rv.values.tolist(), '2minus_uniform_distribution')

    def test_linear_combinations(self):
        rv = 2 * RandomVariable.uniform(0, 1) - RandomVariable.normal(1, 1)
        assert rv.expectation() == 0.02504284382526962
        assert rv.prob_gt(1) == 0.19
        assert rv.prob_lt(.0) == 0.5
        self.assertEqualToCached(
            rv.values.tolist(), 'linear_combination')

    def test_pos_part(self):
        rv = RandomVariable.normal(1, 1, 1000).pos_part()

        assert rv.expectation() == 1.0810155192950992
        assert rv.prob_gt(1) == 0.5
        assert rv.prob_lt(-1) == 0
        self.assertEqualToCached(
            rv.values.tolist(), 'normal_pos_part')


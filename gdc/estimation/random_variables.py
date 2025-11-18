import numpy as np
from scipy.stats import norm, gumbel_r, uniform, lognorm
from scipy.interpolate import interp1d


class RandomVariable:
    def __init__(self, quantiles, values):
        assert len(quantiles) == len(values)
        self.quantiles = np.array(quantiles)
        self.pdf = np.diff(self.quantiles, prepend=self.quantiles[0])
        self.num_quantiles = len(quantiles)
        self.values = np.array(values)
        self.interp = interp1d(self.quantiles, self.values, kind='linear',
                               bounds_error=False,
                               fill_value=(self.values[0], self.values[-1]))

    @classmethod
    def from_distribution(cls, dist, num_quantiles=100):
        quantiles = np.linspace(
            1./num_quantiles, 1-1./num_quantiles, num_quantiles)
        values = dist.ppf(quantiles)
        return cls(quantiles, values)

    @classmethod
    def normal(cls, mu=0, sigma=1, num_quantiles=100):
        return cls.from_distribution(norm(loc=mu, scale=sigma), num_quantiles)

    @classmethod
    def lognormal(cls, mu=0, sigma=1, num_quantiles=100):
        return cls.from_distribution(
            lognorm(s=sigma, scale=np.exp(mu)), num_quantiles)

    @classmethod
    def gumbel(cls, mu=0, beta=1, num_quantiles=100):
        return cls.from_distribution(
            gumbel_r(loc=mu, scale=beta), num_quantiles)

    @classmethod
    def uniform(cls, low=0, high=1, num_quantiles=100):
        return cls.from_distribution(
            uniform(loc=low, scale=high - low), num_quantiles)

    def expectation(self):
        return np.dot(self.values, self.pdf)

    def variance(self):
        mean = self.expectation()
        return np.dot((self.values - mean)**2, self.pdf)

    def std(self):
        return np.sqrt(self.variance())

    def sample(self, size=1, seed=None):
        rng = np.random.default_rng(seed)
        random_quantiles = rng.uniform(0, 1, size)
        return self.interp(random_quantiles)

    def __add__(self, other):
        if isinstance(other, RandomVariable):
            return self.add_rv(other)
        else:
            return RandomVariable(self.quantiles, self.values + other)

    def __neg__(self):
        return RandomVariable(self.quantiles, -self.values)

    def __sub__(self, other):
        if isinstance(other, RandomVariable):
            return self.add_rv(-other)
        else:
            return RandomVariable(self.quantiles, self.values - other)

    def add_rv(self, other):
        combined_values = np.add.outer(self.values, other.values).ravel()
        quantiles = np.linspace(
            1./self.num_quantiles, 1-1./self.num_quantiles, self.num_quantiles)
        values = np.quantile(combined_values, quantiles)
        return RandomVariable(quantiles, values)

    def __mul__(self, scalar):
        return RandomVariable(self.quantiles, self.values * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def pos_part(self):
        return RandomVariable(self.quantiles, np.maximum(self.values, 0))

    def apply(self, func):
        return RandomVariable(self.quantiles, func(self.values))

    def prob_gt(self, threshold):
        return np.mean(self.values > threshold)

    def prob_lt(self, threshold):
        return np.mean(self.values < threshold)


class Operators:
    @classmethod
    def max(cls, list_of_rvs, num_quantiles=100):
        # distribution of the maximum of many random variables
        result = list_of_rvs[0]
        values = result.values
        quantiles = np.linspace(
            1. / num_quantiles, 1 - 1. / num_quantiles,num_quantiles)
        for other in list_of_rvs[1:]:
            combined_values = np.maximum.outer(
                values, other.values).ravel()
            values = np.quantile(combined_values, quantiles)
        return RandomVariable(quantiles, values)

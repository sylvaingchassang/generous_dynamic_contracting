import json
import unittest
import os


class CachedTestCase(unittest.TestCase):

    def _file_path(self, refname):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(base_dir, 'testdb', refname + '.json')

    def save_results(self, results, refname):
        with open(self._file_path(refname), 'w') as f:
            json.dump(results, f)

    def load_results(self, refname):
        filename = self._file_path(refname)
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        else:
            return None

    def assertEqualToCached(self, results, refname, regenerate=False):
        if regenerate:
            self.save_results(results, refname)
        expected = self.load_results(refname)
        assert results == expected

    def _almost_equal_nested(self, actual, expected, places=7, delta=None):
        """
        Recursively compare nested dictionaries with approximate equality
        for floats.

        Args:
            actual: The actual value (can be nested dict, list, or scalar)
            expected: The expected value (can be nested dict, list, or scalar)
            places: Number of decimal places for comparison (default: 7)
            delta: Tolerance for absolute difference (alternative to places)

        Returns:
            bool: True if approximately equal, False otherwise
        """
        # Handle None cases
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False

        # Handle different types
        if type(actual) != type(expected):
            return False

        # Handle dictionaries
        if isinstance(actual, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(
                self._almost_equal_nested(actual[key], expected[key], places,
                                          delta)
                for key in actual.keys()
            )

        # Handle lists/tuples
        if isinstance(actual, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(
                self._almost_equal_nested(a, e, places, delta)
                for a, e in zip(actual, expected)
            )

        # Handle numeric values
        if isinstance(actual, (int, float)) and isinstance(expected,
                                                           (int, float)):
            if delta is not None:
                return abs(actual - expected) <= delta
            else:
                # Use places-based comparison
                return round(abs(actual - expected), places) == 0

        # Handle strings and other types with exact equality
        return actual == expected

    def assertAlmostEqualToCached(self, results, refname, places=7, delta=None,
                                  regenerate=False):
        """
        Compare results to cached values with approximate equality for floating
         point numbers. Supports nested dictionaries and lists.

        Args:
            results: The actual results to compare
            refname: Reference name for the cached file
            places: Number of decimal places for float comparison (default: 7)
            delta: Tolerance for absolute difference (alternative to places)
            regenerate: If True, save new results to cache instead of comparing
        """
        if regenerate:
            self.save_results(results, refname)
            return

        expected = self.load_results(refname)
        if expected is None:
            self.fail(
                f"No cached results found for '{refname}'. "
                f"Run with regenerate=True to create cache.")

        if not self._almost_equal_nested(results, expected, places, delta):
            self.fail(
                f"Results do not approximately match cached values for '{refname}'.\n"
                f"Actual: {results}\n"
                f"Expected: {expected}")




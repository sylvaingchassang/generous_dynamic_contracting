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




import unittest
import os
import pandas as pd
import zipfile
import io
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from gdc.tests.testutils import CachedTestCase
from gdc.cms.data_access import (
    get_zip_files, get_zipfile_metrics, zip_chunk_generator, load_codebook,
    CMS_DATA_PATH, ZF
)
from gdc.utils import ExtendedNamespace


class TestDataAccess(CachedTestCase):
    """Tests for the cms/data_access.py module."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Create a mock CSV file
        self.csv_data = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n"

        # Create a mock ZIP file containing the CSV
        self.zip_path = os.path.join(self.temp_dir, "test_data.zip")
        with zipfile.ZipFile(self.zip_path, 'w') as zf:
            zf.writestr("test_data.csv", self.csv_data)

        # Create a mock codebook CSV
        self.codebook_data = "variable,name\ncol1,Column 1\ncol2,Column 2\ncol3,Column 3\n"
        self.codebook_path = os.path.join(self.temp_dir, "code_book.csv")
        with open(self.codebook_path, 'w') as f:
            f.write(self.codebook_data)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @patch('gdc.cms.data_access.CMS_DATA_PATH')
    @patch('gdc.cms.data_access.os.listdir')
    def test_get_zip_files(self, mock_listdir, mock_cms_data_path):
        """Test the get_zip_files function."""
        # Mock the CMS_DATA_PATH and os.listdir
        mock_cms_data_path.__str__.return_value = self.temp_dir
        mock_listdir.return_value = ["test_data.zip", "not_a_zip.txt"]

        # Call the function
        with patch('gdc.cms.data_access.os.path.join', return_value=os.path.join(self.temp_dir, "test_data.zip")):
            result = get_zip_files()

        # Verify the result has the expected keys
        self.assertIn('test_data', result.__dict__)

        # Convert to dictionaries for caching
        result_dict = {k: v for k, v in result.__dict__.items()}

        # Save results for future test runs
        self.save_results(result_dict, "get_zip_files_result")

    def test_get_zipfile_metrics(self):
        """Test the get_zipfile_metrics function."""
        # Call the function with our test ZIP file
        result = get_zipfile_metrics(self.zip_path)

        # Convert to dictionary for comparison
        result_dict = {k: v for k, v in result.__dict__.items()}

        # Expected result
        expected = {
            'columns': ['col1', 'col2', 'col3'],
            'num_cols': 3,
            'num_rows': 3
        }

        # Verify the result has the expected structure
        self.assertEqual(set(result_dict.keys()), set(expected.keys()))
        self.assertEqual(result_dict['columns'], expected['columns'])
        self.assertEqual(result_dict['num_cols'], expected['num_cols'])
        self.assertEqual(result_dict['num_rows'], expected['num_rows'])

        # Save results for future test runs
        self.save_results(result_dict, "get_zipfile_metrics_result")

    def test_zip_chunk_generator(self):
        """Test the zip_chunk_generator function."""
        # Call the function with our test ZIP file
        with patch('gdc.cms.data_access.CMS_DATA_PATH', self.temp_dir):
            chunks = list(zip_chunk_generator(self.zip_path))

        # We should get one chunk with all the data
        self.assertEqual(len(chunks), 1)

        # Verify the DataFrame has the expected columns and data
        self.assertEqual(set(chunks[0].columns), {'col1', 'col2', 'col3'})
        self.assertEqual(len(chunks[0]), 3)  # 3 rows

        # Convert DataFrame to dictionary for caching
        df_dict = chunks[0].to_dict()

        # Save results for future test runs
        self.save_results(df_dict, "zip_chunk_generator_result")

    def test_zip_chunk_generator_with_batch_size(self):
        """Test the zip_chunk_generator function with batch_size."""
        # Call the function with our test ZIP file and batch_size=1
        with patch('gdc.cms.data_access.CMS_DATA_PATH', self.temp_dir):
            chunks = list(zip_chunk_generator(self.zip_path, batch_size=1))

        # We should get 3 chunks (one for each row)
        self.assertEqual(len(chunks), 3)

        # Verify each chunk has the expected columns and one row
        for i, chunk in enumerate(chunks):
            self.assertEqual(set(chunk.columns), {'col1', 'col2', 'col3'})
            self.assertEqual(len(chunk), 1)  # 1 row per chunk

        # Convert DataFrames to dictionaries for caching
        chunks_dict = [chunk.to_dict() for chunk in chunks]

        # Save results for future test runs
        self.save_results(chunks_dict, "zip_chunk_generator_batched_result")

    def test_zip_chunk_generator_with_max_batches(self):
        """Test the zip_chunk_generator function with max_batches."""
        # Call the function with our test ZIP file, batch_size=1, and max_batches=2
        with patch('gdc.cms.data_access.CMS_DATA_PATH', self.temp_dir):
            chunks = list(zip_chunk_generator(self.zip_path, batch_size=1, max_batches=2))

        # We should get 2 chunks (limited by max_batches)
        self.assertEqual(len(chunks), 2)

        # Verify each chunk has the expected columns and one row
        for i, chunk in enumerate(chunks):
            self.assertEqual(set(chunk.columns), {'col1', 'col2', 'col3'})
            self.assertEqual(len(chunk), 1)  # 1 row per chunk

        # Convert DataFrames to dictionaries for caching
        chunks_dict = [chunk.to_dict() for chunk in chunks]

        # Save results for future test runs
        self.save_results(chunks_dict, "zip_chunk_generator_max_batches_result")

    def test_zip_chunk_generator_with_usecols(self):
        """Test the zip_chunk_generator function with usecols."""
        # Call the function with our test ZIP file and usecols=['col1', 'col3']
        with patch('gdc.cms.data_access.CMS_DATA_PATH', self.temp_dir):
            chunks = list(zip_chunk_generator(self.zip_path, usecols=['col1', 'col3']))

        # We should get one chunk with only the specified columns
        self.assertEqual(len(chunks), 1)
        self.assertEqual(set(chunks[0].columns), {'col1', 'col3'})

        # Verify the DataFrame has the expected number of rows
        self.assertEqual(len(chunks[0]), 3)  # 3 rows

        # Convert DataFrame to dictionary for caching
        df_dict = chunks[0].to_dict()

        # Save results for future test runs
        self.save_results(df_dict, "zip_chunk_generator_usecols_result")

    @patch('gdc.cms.data_access.CMS_DATA_PATH')
    @patch('gdc.cms.data_access.os.path.join')
    @patch('pandas.read_csv')
    def test_load_codebook(self, mock_read_csv, mock_join, mock_cms_data_path):
        """Test the load_codebook function."""
        # Mock the dependencies
        mock_cms_data_path.__str__.return_value = self.temp_dir
        mock_join.return_value = self.codebook_path

        # Create a mock DataFrame for the codebook
        mock_df = pd.DataFrame({
            'variable': ['col1', 'col2', 'col3'],
            'name': ['Column 1', 'Column 2', 'Column 3']
        })
        mock_read_csv.return_value = mock_df

        # Call the function
        result = load_codebook()

        # Convert to dictionary for comparison
        result_dict = {k: v for k, v in result.__dict__.items() if k != '_ExtendedNamespace__dict'}

        # Expected result
        expected = {
            'col1': 'Column 1',
            'col2': 'Column 2',
            'col3': 'Column 3'
        }

        # Verify the result has the expected structure
        self.assertEqual(set(result_dict.keys()), set(expected.keys()))
        for key in expected:
            self.assertEqual(result_dict[key], expected[key])

        # Save results for future test runs
        self.save_results(result_dict, "load_codebook_result")

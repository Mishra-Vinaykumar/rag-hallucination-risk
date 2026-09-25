import sqlite3
import tempfile
import unittest
from pathlib import Path

from build_database import build_database
from validate_database import validate


class DatabaseIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.database = Path(cls.temp_directory.name) / "experiments.sqlite3"
        build_database(cls.project_root, cls.database)

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()

    def test_database_passes_validation(self):
        counts = validate(self.database)
        self.assertEqual(counts["retrieval_events"], 1500)
        self.assertEqual(counts["responses"], 60)

    def test_views_return_expected_rows(self):
        connection = sqlite3.connect(self.database)
        try:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM v_response_labels").fetchone()[0],
                60,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM v_retrieval_analysis").fetchone()[0],
                1200,
            )
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()

"""Integration tests against REAL Polars DataFrames (no fakes/mocks).

Run independently of test_logged_polars.py, whose fake-polars import changes
sys.modules in-process:

    python -m unittest -v test_polars_integration.py

Requirements: Python >= 3.11 recommended; pip install polars
Each test gets its own temporary audit log and small in-memory dataset.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
import warnings

try:
    import polars as pl
except ImportError as exc:
    raise RuntimeError(
        "Real-Polars integration tests require Polars: python -m pip install polars. "
        "Run this file separately from test_logged_polars.py."
    ) from exc

from logged_polars import LoggedDataFrame, configure_logging


class RealPolarsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = pathlib.Path(self.tmp.name)
        self.log_path = self.directory / "processing.log"
        configure_logging(str(self.log_path))
        self.animals = LoggedDataFrame(
            pl.DataFrame(
                {
                    "animal_id": [1, 2, 3, 4, 5],
                    "kind": ["cow", "cow", "pig", "pig", "cow"],
                    "value": [10, 20, 15, 25, 30],
                    "year": [2020, 2021, 2020, 2022, 2022],
                }
            )
        )

    def audit(self):
        # FileHandler writes synchronously; flush explicitly for clarity.
        return self.log_path.read_text(encoding="utf-8")

    def test_real_dataframe_is_wrapped_and_native_is_polars(self):
        self.assertIsInstance(self.animals.native, pl.DataFrame)
        self.assertEqual(self.animals.native.shape, (5, 4))
        self.assertIn("NOTE: OPEN", self.audit())
        self.assertIn("observations=5", self.audit())

    def test_filter_logs_real_row_counts_and_variable_names(self):
        selected = self.animals.filter(pl.col("value") > 15)
        self.assertEqual(selected.native["animal_id"].to_list(), [2, 4, 5])
        text = self.audit()
        self.assertIn("FILTER", text)
        self.assertIn("INPUT: self.animals | rows=5", text)
        self.assertIn("OUTPUT: selected | rows=3", text)
        self.assertIn("row change=-2", text)
        self.assertIn("test_polars_integration.py", text)

    def test_multiline_chain_uses_one_source_block_and_tracks_columns(self):
        result = (
            self.animals
            .filter(pl.col("value") >= 15)
            .with_columns((pl.col("value") * 2).alias("doubled"))
            .select("animal_id", "doubled")
        )
        self.assertEqual(result.native.shape, (4, 2))
        self.assertEqual(result.native["doubled"].to_list(), [40, 30, 50, 60])
        text = self.audit()
        self.assertEqual(text.count("[execution "), 1)
        for operation in ("FILTER", "WITH_COLUMNS", "SELECT"):
            self.assertIn(operation, text)
        self.assertIn("Columns created: doubled", text)
        self.assertIn("Columns removed: kind, value, year", text)
        self.assertIn("OUTPUT: result | rows=4", text)

    def test_repeated_method_on_same_line_stays_one_execution(self):
        result = self.animals.filter(pl.col("value") > 10).filter(pl.col("value") < 30)
        self.assertEqual(result.native.height, 3)
        text = self.audit()
        self.assertEqual(text.count("[execution "), 1)
        self.assertEqual(text.count("NOTE ["), 2)
        self.assertIn("OUTPUT: result | rows=3", text)

    def test_group_by_agg_logs_actual_groups(self):
        totals = self.animals.group_by("kind").agg(pl.col("value").sum().alias("total"))
        self.assertEqual(totals.native.height, 2)
        self.assertEqual(
            dict(zip(totals.native["kind"].to_list(), totals.native["total"].to_list())),
            {"cow": 60, "pig": 40},
        )
        text = self.audit()
        self.assertIn("GROUP_BY.AGG", text)
        self.assertIn("INPUT: self.animals | rows=5", text)
        self.assertIn("OUTPUT: totals | rows=2", text)

    def test_left_join_logs_both_inputs_and_output_name(self):
        pedigree = LoggedDataFrame(
            pl.DataFrame({"animal_id": [1, 2, 2, 4], "parent": ["A", "B", "C", "D"]})
        )
        joined = self.animals.join(pedigree, on="animal_id", how="left")
        self.assertEqual(joined.native.height, 6)
        self.assertEqual(joined.native.filter(pl.col("animal_id") == 2).height, 2)
        text = self.audit()
        self.assertIn("JOIN TYPE: LEFT", text)
        self.assertIn("ON: animal_id", text)
        self.assertIn("INPUT: self.animals | rows=5", text)
        self.assertIn("INPUT 2: pedigree | rows=4", text)
        self.assertIn("OUTPUT: joined | rows=6", text)
        self.assertIn("row change=+1", text)

    def test_join_with_positional_how(self):
        """Real Polars accepts the join type as a keyword; this tests right input unwrapping."""
        other = LoggedDataFrame(pl.DataFrame({"animal_id": [3], "score": [90]}))
        combined = self.animals.join(other, on="animal_id", how="inner")
        self.assertEqual(combined.native.height, 1)
        self.assertIn("INPUT 2: other | rows=1", self.audit())
        self.assertIn("OUTPUT: combined | rows=1", self.audit())

    def test_nested_operations_on_different_inputs_share_source_block(self):
        other = LoggedDataFrame(pl.DataFrame({"animal_id": [1, 4, 5]}))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r".*is_in.*collection.*ambiguous.*",
                category=DeprecationWarning,
            )
            subset = self.animals.filter(
                pl.col("animal_id").is_in(
                    other.select("animal_id").to_series().to_list()
                )
            )
        self.assertEqual(subset.native["animal_id"].to_list(), [1, 4, 5])
        text = self.audit()
        self.assertEqual(text.count("[execution "), 1)
        self.assertLess(text.index("SELECT [line"), text.index("FILTER [line"))
        self.assertIn("OUTPUT: subset | rows=3", text)

    def test_loop_logs_each_iteration_as_separate_execution(self):
        counts = []
        for year in (2020, 2021, 2022):
            slice_df = self.animals.filter(pl.col("year") == year)
            counts.append(slice_df.native.height)
        self.assertEqual(counts, [2, 1, 2])
        self.assertEqual(self.audit().count("[execution "), 3)

    def test_same_line_separate_statements_have_different_executions(self):
        a = self.animals.filter(pl.col("value") > 10); b = self.animals.select("kind")
        self.assertEqual(a.native.height, 4)
        self.assertEqual(b.native.height, 5)
        self.assertEqual(self.audit().count("[execution "), 2)

    def test_helper_function_reports_helper_source(self):
        def get_pigs(df):
            return df.filter(pl.col("kind") == "pig")
        pigs = get_pigs(self.animals)
        self.assertEqual(pigs.native.height, 2)
        text = self.audit()
        self.assertIn('return df.filter(pl.col("kind") == "pig")', text)
        self.assertIn("INPUT: df | rows=5", text)

    def test_polars_exception_is_logged_and_reraised(self):
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            self.animals.filter(pl.col("no_such_column") > 0)
        text = self.audit()
        self.assertIn("ERROR [", text)
        self.assertIn("FILTER FAILED", text)
        self.assertIn("no_such_column", text)
        self.assertIn("[execution ", text)

    def test_write_parquet_logs_request_and_creates_file(self):
        output_path = self.directory / "output.parquet"
        self.animals.write_parquet(output_path)
        self.assertTrue(output_path.exists())
        self.assertEqual(pl.read_parquet(output_path).height, 5)
        text = self.audit()
        self.assertIn("WRITE_PARQUET", text)
        self.assertIn("Write requested from self.animals for 5 rows", text)

    def test_configure_logging_resets_file_between_runs(self):
        self.animals.select("animal_id")
        self.assertIn("SELECT", self.audit())
        configure_logging(str(self.log_path))
        self.assertEqual(self.audit(), "")
        self.animals.filter(pl.col("value") > 10)
        self.assertNotIn("SELECT", self.audit())
        self.assertIn("FILTER", self.audit())


if __name__ == "__main__":
    unittest.main()

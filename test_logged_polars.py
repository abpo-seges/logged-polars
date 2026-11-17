"""Self-contained source-tracker regression tests using simulated Polars objects.

Run: python -m unittest -v test_logged_polars.py
These tests are not substitutes for integration tests with real Polars.
"""
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

HERE = pathlib.Path(__file__).parent
p = types.ModuleType('polars')

class FakeDataFrame:
    def __init__(self, n=8, columns=('value',)):
        self.height = n
        self.columns = list(columns)
        self.width = len(columns)

    def filter(self, condition):
        if condition == 'fail':
            raise ValueError('deliberate failure')
        return FakeDataFrame(max(0, self.height-1), self.columns)

    def select(self, *args):
        return FakeDataFrame(self.height, self.columns)

    def with_columns(self, *args):
        return FakeDataFrame(self.height, [*self.columns, 'extra'])

    def join(self, other, **kwargs):
        return FakeDataFrame(self.height + other.height, self.columns)

    def group_by(self, *args):
        return FakeGroupBy(self)

class FakeGroupBy:
    def __init__(self, parent):
        self.parent = parent

    def agg(self, *args):
        return FakeDataFrame(2, self.parent.columns)

p.DataFrame = FakeDataFrame
sys.modules['polars'] = p
spec = importlib.util.spec_from_file_location('logged_polars', HERE / 'logged_polars.py')
module = importlib.util.module_from_spec(spec)
sys.modules['logged_polars'] = module
spec.loader.exec_module(module)

class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = pathlib.Path(self.tmp.name) / 'audit.log'
        module.configure_logging(str(self.log))
        self.df = module.LoggedDataFrame(FakeDataFrame(), 'left')

    def content(self):
        for handler in module._LOG.handlers:
            handler.flush()
        return self.log.read_text()

    def test_multiline_chained_and_groupby(self):
        result = (
            self.df
            .filter(True)
            .with_columns(True)
            .group_by('value')
            .agg('extra')
        )
        log = self.content()
        self.assertEqual(log.count('[execution '), 1)
        for name in ('FILTER', 'WITH_COLUMNS', 'GROUP_BY.AGG'):
            self.assertIn(name, log)
        self.assertEqual(result.native.height, 2)

    def test_repeated_method_same_line(self):
        result = self.df.filter(True).filter(True)
        self.assertEqual(self.content().count('[execution '), 1)
        self.assertEqual(self.content().count('NOTE ['), 2)
        self.assertEqual(result.native.height, 6)

    def test_loop_repeats_source(self):
        for i in range(3):
            unused = self.df.filter(i)
        self.assertEqual(self.content().count('[execution '), 3)

    def test_nested_separate_dataframes(self):
        right = module.LoggedDataFrame(FakeDataFrame(), 'right')
        result = self.df.filter(right.select('value'))
        log = self.content()
        self.assertEqual(log.count('[execution '), 1)
        self.assertLess(log.index('SELECT [line'), log.index('FILTER [line'))

    def test_branch_and_helper(self):
        def helper(df, enabled):
            if enabled:
                return df.filter(True)
            return df.select('value')
        helper(self.df, True)
        helper(self.df, False)
        log = self.content()
        self.assertEqual(log.count('[execution '), 2)
        self.assertIn('test_logged_polars.py', log)

    def test_exception_has_source(self):
        with self.assertRaisesRegex(ValueError, 'deliberate failure'):
            self.df.filter('fail')
        log = self.content()
        self.assertIn('ERROR [', log)
        self.assertIn("self.df.filter('fail')", log)

    def test_same_line_different_statements(self):
        a = self.df.filter(True); b = self.df.select('value')
        self.assertEqual(self.content().count('[execution '), 2)

    def test_variable_names_for_join(self):
        animals = module.LoggedDataFrame(FakeDataFrame(10))
        pedigree = module.LoggedDataFrame(FakeDataFrame(4))
        result = animals.join(pedigree, on='value', how='left')
        log = self.content()
        self.assertIn('INPUT: animals | rows=10', log)
        self.assertIn('INPUT 2: pedigree | rows=4', log)
        self.assertIn('OUTPUT: result | rows=14', log)
        self.assertIn('JOIN TYPE: LEFT', log)

    def test_chained_operations_output_name(self):
        animals = module.LoggedDataFrame(FakeDataFrame(10))
        selected = animals.filter(True).with_columns(True)
        log = self.content()
        self.assertIn('INPUT: animals | rows=10', log)
        self.assertIn('OUTPUT: <intermediate>', log)
        self.assertIn('OUTPUT: selected | rows=9', log)
        self.assertEqual(selected.native.height, 9)

    def test_reassigned_input_variable(self):
        animals = module.LoggedDataFrame(FakeDataFrame(10))
        animals = animals.filter(True)
        log = self.content()
        self.assertIn('INPUT: animals | rows=10', log)
        self.assertIn('OUTPUT: animals | rows=9', log)

    def test_join_logs_right_input(self):
        right = module.LoggedDataFrame(FakeDataFrame(3), 'right')
        combined = self.df.join(right, on='value')
        self.assertEqual(combined.native.height, 11)
        self.assertIn('INPUT 2: right | rows=3', self.content())
        self.assertIn('OUTPUT: combined | rows=11', self.content())
        self.assertIn('INPUT: self.df | rows=8', self.content())

if __name__ == '__main__':
    unittest.main()

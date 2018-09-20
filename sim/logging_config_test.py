import unittest as ut
import configparser as cp
import logging_config as lc
import logging

class TestLoggingConfig(ut.TestCase):

    def setUp(self):
        self._original_config = cp.ConfigParser()
        self._original_config.read('./test_config.ini')
        self._logging_config = lc.LoggingConfig(self._original_config['Logging'])

    def test_get_logging_level(self):
        self.assertEqual(self._logging_config.get_logging_level(), logging.DEBUG)

    def test_get_logging_level_empty(self):
        self._original_config.set('Logging', 'level', '')
        self.assertEqual(self._logging_config.get_logging_level(), logging.ERROR)

    def test_get_logging_level_wrong_setting(self):
        self._original_config.set('Logging', 'level', 'wrong-level')
        self.assertEqual(self._logging_config.get_logging_level(), logging.ERROR)

    def test_get_location(self):
        self.assertEqual(self._logging_config.get_location(), 'log')

    def test_get_test_result_location(self):
        self.assertEqual(self._logging_config.get_test_result_location(), 'test_results')

    def test_get_logfile_prefix(self):
        self.assertEqual(self._logging_config.get_logfile_prefix(), 'log_')

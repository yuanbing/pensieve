import logging

class LoggingConfig:

    def __init__(self, raw_config):
        self._raw_config = raw_config

        self._logging_level_map = {
            'critical': logging.CRITICAL,
            'fatal': logging.fatal,
            'error': logging.ERROR,
            'warn': logging.WARN,
            'warning': logging.WARN,
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'notset': logging.NOTSET
        }

        self._default_logging_level = 'error'

    def get_logging_level(self):
        current_logging_level = self._raw_config.get('level')
        if current_logging_level:
            current_logging_level = current_logging_level.lower()
        else:
            current_logging_level = self._default_logging_level

        if current_logging_level in self._logging_level_map.keys():
            return self._logging_level_map[current_logging_level]
        else:
            return self._logging_level_map[self._default_logging_level]

    def get_location(self):
        return self._raw_config.get('location')

    def get_test_result_location(self):
        return self._raw_config.get('test_result_location')

    def get_logfile_prefix(self):
        return self._raw_config.get('logfile_prefix')

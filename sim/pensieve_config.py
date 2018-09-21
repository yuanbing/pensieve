import configparser
import model_saving_config as msc
import model_training_config as mtc
import logging_config as lc


class PensieveTrainingConfig:

    def __init__(self, config_file):
        self._config = configparser.ConfigParser()
        self._config.read(config_file)
        self._model_saving_config = msc.ModelSavingConfig(self._config['Model Saving'])
        self._model_training_config = mtc.ModelTrainingConfig(self._config['Model Training'])
        self._logging_config = lc.LoggingConfig(self._config['Logging'])

    def get_model_saving_config(self):
        """
        :return: instance of ModelSavingConfig
        """
        return self._model_saving_config

    def get_model_training_config(self):
        """
        :return: instance of ModelTrainingConfig
        """
        return self._model_training_config

    def get_logging_config(self):
        """
        :return: instance of LoggingConfig
        """
        return self._logging_config


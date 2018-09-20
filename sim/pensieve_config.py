import configparser
import model_saving_config as msc
import model_training_config as mtc


class PensieveTrainingConfig:

    def __init__(self, config_file):
        self._config = configparser.ConfigParser()
        self._config.read(config_file)

    def get_model_saving_config(self):
        """
        Returns an instance of ModelSavingConfig
        :return: instance of ModelSavingConfig
        """
        return msc.ModelSavingConfig(self._config['Model Saving'])

    def get_model_training_config(self):
        return mtc.ModelTrainingConfig(self._config['Model Training'])


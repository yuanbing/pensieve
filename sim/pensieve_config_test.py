import unittest
import pensieve_config as tc


class TestPensieveConfig(unittest.TestCase):

    def setUp(self):
        self._test_config = tc.PensieveTrainingConfig('test_config.ini')

    def test_loading_config(self):
        self.assertIsNotNone(self._test_config)

    def test_has_training_config(self):
        self.assertIsNotNone(self._test_config.get_model_saving_config())

    def test_training_config(self):
        model_saving_config = self._test_config.get_model_saving_config()
        self.assertEqual(model_saving_config.get_location(), 'model location')
        self.assertEqual(model_saving_config.get_model_tag(), 'model tag')
        self.assertEqual(model_saving_config.get_inference_method_signature(), 'prediction signature')
        self.assertEqual(model_saving_config.get_inference_method_name(), 'prediction')
        self.assertEqual(model_saving_config.get_inference_method_input(), 'prediction input')
        self.assertEqual(model_saving_config.get_inference_method_output(), 'prediction output')


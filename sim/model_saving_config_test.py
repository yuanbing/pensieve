import unittest
from model_saving_config import ModelSavingConfig


class TestModelSavingConfig(unittest.TestCase):

    def setUp(self):
        self.test_config_with_tag = ModelSavingConfig({
            'location': 'a fake location',
            'tag': 'some tag',
            'inference': 'some inference method',
            'input': 'inference input',
            'output': 'inference output'
        })

        self.test_config_without_tag = ModelSavingConfig({
            'location': 'a fake location',
            'inference': 'some inference method',
            'input': 'inference input',
            'output': 'inference output'
        })

    def test_model_saving_config_with_tag(self):
        self.assertEqual(self.test_config_with_tag.get_location(), 'a fake location')
        self.assertEqual(self.test_config_with_tag.get_model_tag(), 'some tag')
        self.assertEqual(self.test_config_with_tag.get_inference_method_name(), 'some inference method')
        self.assertEqual(self.test_config_with_tag.get_inference_method_input(), 'inference input')
        self.assertEqual(self.test_config_with_tag.get_inference_method_output(), 'inference output')

    def test_model_saving_config_without_tag(self):
        self.assertEqual(self.test_config_without_tag.get_location(), 'a fake location')
        self.assertEqual(self.test_config_without_tag.get_model_tag(), 'actor_model')
        self.assertEqual(self.test_config_without_tag.get_inference_method_name(), 'some inference method')
        self.assertEqual(self.test_config_without_tag.get_inference_method_input(), 'inference input')
        self.assertEqual(self.test_config_without_tag.get_inference_method_output(), 'inference output')


if __name__ == '__main__':
    unittest.main()
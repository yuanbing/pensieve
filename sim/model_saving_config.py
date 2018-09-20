class ModelSavingConfig:

    def __init__(self, raw_config):
        assert(raw_config is not None)
        self._raw_config = raw_config

    def get_location(self):
        return self._raw_config.get('location')

    def get_model_tag(self):
        return self._raw_config.get('tag', 'actor_model')

    def get_inference_method_name(self):
        return self._raw_config.get('inference')

    def get_inference_method_signature(self):
        return self._raw_config.get('signature')

    def get_inference_method_input(self):
        return self._raw_config.get('input')

    def get_inference_method_output(self):
        return self._raw_config.get('output')

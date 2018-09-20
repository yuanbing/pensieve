import unittest as ut
import configparser as cp
import model_training_config as mtc
import multiprocessing as mp


class TestModelTrainingConfig(ut.TestCase):

    def setUp(self):
        self._original_config = cp.ConfigParser()
        self._original_config.read('./test_config.ini')
        self._training_config = mtc.ModelTrainingConfig(self._original_config['Model Training'])

    def test_training_loaded(self):
        self.assertIsNotNone(self._training_config)

    def test_get_states(self):
        self.assertEqual(self._training_config.get_states(), 3)

    def test_get_state_history(self):
        self.assertEqual(self._training_config.get_state_history(), 4)

    def test_get_actions(self):
        self.assertEqual(self._training_config.get_actions(), 2)

    def test_get_actor_learning_rate(self):
        self.assertEqual(self._training_config.get_actor_learning_rate(), 0.01)

    def test_get_critic_learning_rate(self):
        self.assertEqual(self._training_config.get_critic_learning_rate(), 0.01)

    def test_number_of_agents(self):
        self.assertEqual(self._training_config.get_number_of_agents(), 2)

    def test_number_of_agents_not_preset(self):
        self._original_config.set('Model Training', 'number_of_agents', '-1')
        self.assertEqual(self._training_config.get_number_of_agents(), mp.cpu_count()*2)

    def test_get_batch_size(self):
        self.assertEqual(self._training_config.get_batch_size(), 10)

    def test_get_bitrates(self):
        bitrates = self._training_config.get_bitrates()
        self.assertEqual(len(bitrates), 3)
        self.assertEqual(bitrates, [300, 750, 1200])

    def test_hd_reward(self):
        hd_rewards = self._training_config.get_hd_reward()
        self.assertEqual(len(hd_rewards), 3)
        self.assertEqual(hd_rewards, [1, 2, 3])

    def test_buffer_norm_factor(self):
        self.assertEqual(self._training_config.get_buffer_norm_factor(), 10.0)

    def test_get_chunk_cap(self):
        self.assertEqual(self._training_config.get_chunk_cap(), 24.0)

    def test_get_rebuffer_penalty(self):
        self.assertEqual(self._training_config.get_rebuffer_penalty(), 1.5)

    def test_get_smooth_penalty(self):
        self.assertEqual(self._training_config.get_smooth_penalty(), 0.8)

    def get_default_video_quality(self):
        self.assertEqual(self._training_config.get_default_video_quality(), 2)

    def get_max_training_epoch(self):
        self.assertEqual(self._training_config.get_max_training_epoch(), 100)

    def get_network_trace_location(self):
        self.assertEqual(self._training_config.get_network_trace_location(), 'traces')



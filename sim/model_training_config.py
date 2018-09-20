import multiprocessing as mp


class ModelTrainingConfig:

    def __init__(self, raw_config):
        self._raw_config = raw_config

    def get_states(self):
        return self._raw_config.getint('states')

    def get_state_history(self):
        return self._raw_config.getint('state_history')

    def get_actions(self):
        return self._raw_config.getint('actions')

    def get_actor_learning_rate(self):
        return self._raw_config.getfloat('actor_learning_rate')

    def get_critic_learning_rate(self):
        return self._raw_config.getfloat('critic_learning_rate')

    def get_number_of_agents(self):
        number_of_agents = self._raw_config.getint('number_of_agents')

        if number_of_agents == -1:
            number_of_agents = mp.cpu_count()*2

        return number_of_agents

    def get_batch_size(self):
        return self._raw_config.getint('batch_size')

    def get_bitrates(self):
        bitrates_as_string = self._raw_config.get('bitrates')
        return [int(x) for x in bitrates_as_string.split(',')]

    def get_hd_reward(self):
        rewards_as_string = self._raw_config.get('hd_reward')
        return [int(x) for x in rewards_as_string.split(',')]

    def get_buffer_norm_factor(self):
        return self._raw_config.getfloat('buffer_norm_factor')

    def get_chunk_cap(self):
        return self._raw_config.getfloat('chunk_cap')

    def get_rebuffer_penalty(self):
        return self._raw_config.getfloat('rebuffer_penalty')

    def get_smooth_penalty(self):
        return self._raw_config.getfloat('smooth_penalty')

    def get_default_video_quality(self):
        return self._raw_config.getint('default_video_quality')

    def get_max_training_epoch(self):
        return self._raw_config.getint('max_epch')

    def get_network_trace_location(self):
        return self._raw_config.get('network_traces')

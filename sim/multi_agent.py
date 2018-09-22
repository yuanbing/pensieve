import os
import logging
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import env
import a3c
import load_trace
import shutil
import pensieve_config as pc

os.environ['CUDA_VISIBLE_DEVICES'] = ''

RANDOM_SEED = 42
RAND_RANGE = 1000
M_IN_K = 1000.0


def get_reward_function(config):
    bitrates = config.get_bitrates()

    # -- linear reward --
    # reward is video quality - rebuffer penalty - smoothness
    def linear_reward(bitrate, last_bitrate, playback_state):
        return bitrates[bitrate]/M_IN_K - \
               config.get_rebuffer_penalty()*playback_state.rebuffer_time - \
               config.get_smooth_penalty() * np.abs(bitrates[bitrate]-bitrates[last_bitrate])/M_IN_K

    # -- log scale reward --
    # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
    # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

    # reward = log_bit_rate \
    #          - REBUF_PENALTY * rebuf \
    #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

    # -- HD reward --
    # reward = HD_REWARD[bit_rate] \
    #          - REBUF_PENALTY * rebuf \
    #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

    return linear_reward


def test_model(epoch, nn_model, log_file, best_result_so_far, config):
    """
    It tests the saved model

    :param epoch: the epoch index
    :param nn_model: the location of saved model
    :param log_file: the log file handle to save the test results
    :return: None
    """
    # run test script
    os.system('python rl_test_sm.py ' + nn_model)

    # append test performance to the log
    rewards = []
    test_result_path = config.get_test_result_path()
    test_log_files = os.listdir(test_result_path)
    for test_log_file in test_log_files:
        reward = []
        with open(os.path.join(test_result_path, test_log_file), 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write("{0:.0f}\t{1:f}\t{2:f}\t{3:f}\t{4:f}\t{5:f}\t{6:f}\n".format(
        epoch,
        rewards_min,
        rewards_5per,
        rewards_mean,
        rewards_median,
        rewards_95per,
        rewards_max
    ))

    log_file.flush()

    # pick the best trained model so far (based on 95 percentile of rewards
    if len(best_result_so_far) == 0:
        best_result_so_far.append(rewards_95per)
        best_result_so_far.append(nn_model)
    elif best_result_so_far[0] >= rewards_95per:
        shutil.rmtree(nn_model, ignore_errors=True)
    else:
        shutil.rmtree(best_result_so_far[1], ignore_errors=True)
        best_result_so_far[0] = rewards_95per
        best_result_so_far[1] = nn_model


def central_agent(net_params_queues, exp_queues, config):
    """
    it represents the RL training coordinator that aggregates the experiences from workers and updates the NN parameters

    :param net_params_queues: outbound queue for sending NN parameters to worker agents
    :param exp_queues: inbound queue for receiving experiences from workers in training
    :param config: pensieve configuration
    :return: None
    """
    best_result_so_far = []

    log_path = config.get_log_path()

    logging.basicConfig(filename=os.path.join(log_path, 'log_central'),
                        filemode='w',
                        level=config.get_logging_config().get_logging_level())

    with tf.Session(graph=tf.Graph()) as sess, \
            open(os.path.join(log_path, 'log_test'), 'w') as test_log_file:

        training_config = config.get_model_training_config()
        model_saving_config = config.get_model_saving_config()

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[training_config.get_states(), training_config.get_state_history()],
                                 action_dim=training_config.get_actions(),
                                 learning_rate=training_config.get_actor_learning_rate())
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[training_config.get_states(), training_config.get_state_history()],
                                   action_dim=training_config.get_actions(),
                                   learning_rate=training_config.get_critic_learning_rate())

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_path, sess.graph)  # training monitor

        epoch = 0

        number_of_agents = training_config.get_number_of_agents()

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(number_of_agents):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in range(number_of_agents):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert number_of_agents == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            for actor_gradients, critic_gradients in zip(actor_gradient_batch, critic_gradient_batch):
                actor.apply_gradients(actor_gradients)
                critic.apply_gradients(critic_gradients)

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: {} TD_loss: {} Avg_reward: {} Avg_entropy: {}'.
                         format(epoch,
                                avg_td_loss,
                                avg_reward,
                                avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % training_config.get_batch_size() == 0:
                saved_model_location = config.get_model_path(str(epoch))
                logging.info('Saving actor model to location: ' + saved_model_location)
                saver = tf.saved_model.builder.SavedModelBuilder(saved_model_location)
                actor_model_input = {model_saving_config.get_inference_method_input():
                                         tf.saved_model.utils.build_tensor_info(actor.inputs)}
                actor_model_output = {model_saving_config.get_inference_method_output():
                                          tf.saved_model.utils.build_tensor_info(actor.out)}
                actor_model_prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                    actor_model_input,
                    actor_model_output,
                    model_saving_config.get_inference_method_name()
                )
                saver.add_meta_graph_and_variables(sess,
                                                   [model_saving_config.get_model_tag()],
                                                   {
                                                       model_saving_config.get_inference_method_signature():
                                                           actor_model_prediction_signature
                                                   })
                saver.save()

                logging.info('Actor model has been saved to ' + saved_model_location)
                logging.info('Testing saved actor model')
                test_model(epoch, saved_model_location, test_log_file, best_result_so_far, config)

            if epoch == training_config.get_max_training_epoch():
                # tell each worker to shut down
                for i in range(number_of_agents):
                    logging.debug('Informing worker {} to shutdown'.format(i))
                    net_params_queues[i].put(None)
                    #print('Waiting for worker {} to shutdown'.format(i))
                    logging.debug('Waiting for worker {} to shutdown'.format(i))
                    net_params_queues[i].join()
                    logging.debug('Worker {} has shutdown'.format(i))
                break

    logging.debug("Central agent exits")
    return


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, reward_function, config):
    """
    Launch the worker agent

    :param agent_id: str, worker id
    :param all_cooked_time: List[List[float]], network bw timestamp
    :param all_cooked_bw: List[List[float]], network bandwidth
    :param net_params_queue: inbound IPC channel for receiving NN parameters from central agent (coordinator)
    :param exp_queue: outbound IPC channel for sending NN parameters to central agent (coordinator)
    :param reward_function: the reward function
    :param config: pensieve configuration
    :return: None
    """
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)
    log_file = os.path.join(config.get_log_path(), 'log_agent_{}'.format(agent_id))
    training_config = config.get_model_training_config()
    state_dim = training_config.get_states()
    state_history = training_config.get_state_history()

    with tf.Session() as sess, open(log_file, 'w') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[state_dim, state_history],
                                 action_dim=training_config.get_actions(),
                                 learning_rate=training_config.get_actor_learning_rate())

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[state_dim, state_history],
                                   action_dim=training_config.get_actions(),
                                   learning_rate=training_config.get_critic_learning_rate())

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        net_params_queue.task_done()

        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = training_config.get_default_video_quality()
        bit_rate = last_bit_rate

        action_vec = np.zeros(training_config.get_actions())
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((state_dim, state_history))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0

        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            playback_state = net_env.get_video_chunk(bit_rate)

            delay = playback_state.delay
            sleep_time = playback_state.sleep_time
            buffer_size = playback_state.current_buffer_size
            rebuf = playback_state.rebuffer_time
            video_chunk_size = playback_state.video_chunk_size
            next_video_chunk_sizes = playback_state.next_video_chunk_sizes
            end_of_video = playback_state.end_of_video
            video_chunk_remain = playback_state.remain_video_chunks

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            reward = reward_function(bit_rate, last_bit_rate, playback_state)

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((state_dim, state_history))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # deque history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            # last quality
            state[0, -1] = training_config.get_bitrates()[bit_rate] / float(np.max(training_config.get_bitrates()))
            state[1, -1] = buffer_size / training_config.get_buffer_norm_factor()  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / training_config.get_buffer_norm_factor()  # 10 sec
            state[4, :training_config.get_actions()] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, training_config.get_chunk_cap()) / training_config.get_chunk_cap()

            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, state_dim, state_history)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discredit the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write("{0:.0f}\t{1:.0f}\t{2:f}\t{3:f}\t{4:f}\t{5:.0f}\t{6:f}\n".format(
                time_stamp,
                training_config.get_bitrates()[bit_rate],
                buffer_size,
                rebuf,
                video_chunk_size,
                delay,
                reward
            ))

            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= training_config.get_batch_size() or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                net_params = net_params_queue.get()
                if net_params is None:
                    # received signal from central agent to shutdown
                    #print('agent {} is told to shut down'.format(agent_id))
                    logging.debug('Worker {} is told to shut down'.format(agent_id))
                    net_params_queue.task_done()
                    break

                net_params_queue.task_done()
                actor_net_params, critic_net_params = net_params
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = training_config.get_default_video_quality()
                bit_rate = last_bit_rate  # use the default action here

                action_vec = np.zeros(training_config.get_actions())
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((state_dim, state_history)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(training_config.get_actions())
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)

        logging.debug('Worker {} exits'.format(agent_id))
        return


def init(config_file):
    config = pc.PensieveConfig(config_file)

    # purge saved model directory
    saved_model_path = config.get_model_path(None)
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)

    # create the directory for log files if it doesn't exist already
    log_path = config.get_log_path()
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # create the directory for test results if it doesn't exist already
    test_result_path = config.get_test_result_path()
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    return config


def main():
    np.random.seed(RANDOM_SEED)

    config = init('config.ini')

    number_of_agents = config.get_model_training_config().get_number_of_agents()

    # create IPC queues
    net_params_queues = [mp.JoinableQueue(1) for i in range(number_of_agents)]
    exp_queues = [mp.Queue(1) for i in range(number_of_agents)]

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues, config))
    coordinator.start()

    # load trace files
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(config.get_network_trace_path())

    agents = []
    for i in range(number_of_agents):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i],
                                       get_reward_function(config.get_model_training_config()),
                                       config)))
    for i in range(number_of_agents):
        agents[i].start()

    # wait unit training is done
    coordinator.join()

    logging.info('The model training has ended')


if __name__ == '__main__':
    main()

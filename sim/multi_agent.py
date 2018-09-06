import os
import logging
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import env
import a3c
import load_trace

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# TF saved_model directory
ACTOR_MODEL_LOCATION = './trained_actor_model'
ACTOR_MODEL_TAG = 'actor_model'
ACTOR_MODEL_PREDICTION_METHOD_NAME = 'actor_model_prediction'
ACTOR_MODEL_PREDICTION_SIGNATURE_KEY = 'actor_model_prediction_signature_key'
ACTOR_MODEL_INPUT = 'actor_model_input'
ACTOR_MODEL_OUTPUT = 'actor_model_output'

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None


def get_reward_function():
    # -- linear reward --
    # reward is video quality - rebuffer penalty - smoothness
    def linear_reward(bitrate, last_bitrate, playback_state):
        return VIDEO_BIT_RATE[bitrate]/M_IN_K - \
               REBUF_PENALTY*playback_state.rebuffer_time - \
               SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bitrate]-VIDEO_BIT_RATE[last_bitrate])/M_IN_K

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


def testModel(epoch, nn_model, log_file):
    """
    It tests the saved model

    :param epoch: the epoch index
    :param nn_model: the location of saved model
    :param log_file: the log file handle to save the test results
    :return: None
    """
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)

    # run test script
    os.system('python rl_test_sm.py ' + nn_model)

    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:
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


def central_agent(net_params_queues, exp_queues):
    """
    it represents the RL training coordinator that aggregates the experiences from workers and updates the NN parameters

    :param net_params_queues: outbound queue for sending NN parameters to worker agents
    :param exp_queues: inbound queue for receiving experiences from workers in training
    :return: None
    """
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session() as sess, open(LOG_FILE + '_test', 'w') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        # saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        # nn_model = NN_MODEL
        # if nn_model is not None:  # nn_model is the path to file
        #    saver.restore(sess, nn_model)
        #    print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
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

            for i in range(NUM_AGENTS):
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
            assert NUM_AGENTS == len(actor_gradient_batch)
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

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the actor model along with all the weights
                saved_model_location = '{}/{}'.format(ACTOR_MODEL_LOCATION, epoch)
                logging.info('Saving actor model to location: ' + saved_model_location)
                saver = tf.saved_model.builder.SavedModelBuilder(saved_model_location)
                actor_model_input = {ACTOR_MODEL_INPUT: tf.saved_model.utils.build_tensor_info(actor.inputs)}
                actor_model_output = {ACTOR_MODEL_OUTPUT: tf.saved_model.utils.build_tensor_info(actor.out)}
                actor_model_prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                    actor_model_input,
                    actor_model_output,
                    ACTOR_MODEL_PREDICTION_METHOD_NAME
                )
                saver.add_meta_graph_and_variables(sess,
                                                   [ACTOR_MODEL_TAG],
                                                   {
                                                       ACTOR_MODEL_PREDICTION_SIGNATURE_KEY: actor_model_prediction_signature}
                                                   )
                saver.save()

                logging.info('Actor model has been saved to ' + saved_model_location)
                logging.info('Testing saved actor model')
                testModel(epoch, saved_model_location, test_log_file)


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, reward_function):
    """
    Launch the worker agent

    :param agent_id: str, worker id
    :param all_cooked_time: List[List[float]], network bw timestamp
    :param all_cooked_bw: List[List[float]], network bandwidth
    :param net_params_queue: inbound IPC channel for receiving NN parameters from central agent (coordinator)
    :param exp_queue: outbound IPC channel for sending NN parameters to central agent (coordinator)
    :param reward_function: the reward function
    :return: None
    """
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with tf.Session() as sess, open('{}_agent_{}'.format(LOG_FILE, agent_id), 'w') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
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
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # deque history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discredit the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write("{0:.0f}\t{1:.0f}\t{2:f}\t{3:f}\t{4:f}\t{5:.0f}\t{6:f}\n".format(
                time_stamp,
                VIDEO_BIT_RATE[bit_rate],
                buffer_size,
                rebuf,
                video_chunk_size,
                delay,
                reward
            ))

            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def main():
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    # load trace files
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i],
                                       get_reward_function())))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()

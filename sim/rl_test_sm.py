import os
import sys
import numpy as np
import tensorflow as tf
import load_trace
import fixed_env as env
import a3c

# Disable the GPU devices.
# If GPU acceleration is desired, please comment out the following line
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# TF saved_model directory
ACTOR_MODEL_LOCATION = sys.argv[1]
ACTOR_MODEL_TAG = 'actor_model'
ACTOR_MODEL_PREDICTION_METHOD_NAME = 'actor_model_prediction'
ACTOR_MODEL_PREDICTION_SIGNATURE_KEY = 'actor_model_prediction_signature_key'
ACTOR_MODEL_INPUT = 'actor_model_input'
ACTOR_MODEL_OUTPUT = 'actor_model_output'

# TODO: the following section needs to be refactored to a separate module
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './cooked_test_traces/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward


def predict(session, model_output, model_input, x):
    return session.run(model_output, feed_dict={model_input: x})


def main():
    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # Check whether the saved actor model exists
        if not tf.saved_model.loader.maybe_saved_model_directory(ACTOR_MODEL_LOCATION):
            print('Cannot find saved actor model at ' + ACTOR_MODEL_LOCATION)
            sys.exit(1)
        
        # Restore the saved actor model
        saved_actor_model = tf.saved_model.loader.load(sess, [ACTOR_MODEL_TAG], ACTOR_MODEL_LOCATION)
        signature = saved_actor_model.signature_def
        actor_model_input_name = signature[ACTOR_MODEL_PREDICTION_SIGNATURE_KEY].inputs[ACTOR_MODEL_INPUT].name
        actor_model_output_name = signature[ACTOR_MODEL_PREDICTION_SIGNATURE_KEY].outputs[ACTOR_MODEL_OUTPUT].name
        actor_model_input = sess.graph.get_tensor_by_name(actor_model_input_name)
        actor_model_output = sess.graph.get_tensor_by_name(actor_model_output_name)

        #predict = lambda x: sess.run(actor_model_output, feed_dict={actor_model_input: x})

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write('{0:.0f}\t{1:.0f}\t{2:f}\t{3:f}\t{4:f}\t{5:.0f}\t{6:f}\n'.format(
                time_stamp/M_IN_K,
                VIDEO_BIT_RATE[bit_rate],
                buffer_size,
                rebuf,
                video_chunk_size,
                delay,
                reward
            ))
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob = predict(sess, actor_model_output, actor_model_input, np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discredit the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write('\n')
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names): # all the network traces have been experienced
                    break

                log_file.close()

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()

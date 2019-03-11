import argparse
import time
import os
from baselines import logger
from baselines.common.misc_util import (
    set_global_seeds,
)
import baselines.ddpg.training as training
import baselines.ddpg.demo as demo
from baselines.ddpg.config import DEFAULT_PARAMS, get_convert_arg_to_type_fn, prepare_params, log_params, configure_dims, \
    configure_rollout_worker, create_agents, configure_memory
from baselines.ddpg.policy_selection import get_policy_fn
import baselines.common.tf_util as U

import tensorflow as tf
from mpi4py import MPI


def run(args):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # If we are supposed to divide gpu usage among a specific set of devices,
    # set this processes' device to the correct one.
    gpu_nums = args['split_gpu_usage_among_device_nums']
    if gpu_nums is not None:
        gpu_num_to_use = gpu_nums[rank % len(gpu_nums)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num_to_use)

    # Seed everything to make things reproducible.
    rank_seed = args['seed'] + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, rank_seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(rank_seed)

    input_dims = configure_dims(args)

    # Configure the replay buffer.
    memory = configure_memory(args)

    with U.single_threaded_session() as sess:
        # Setup up DDPG Agents

        agents = create_agents(sess=sess, memory=memory, input_dims=input_dims, params=args)

        saver = tf.train.Saver()
        if args['restore_from_ckpt'] is not None:
            logger.info("Restoring agents from {}".format(args['restore_from_ckpt']))
            saver.restore(sess, args['restore_from_ckpt'])

        sess.graph.finalize()
        logger.log_graph_to_tensorboard(sess.graph)

        # Setup Rollout workers
        train_policy_fn = get_policy_fn(
            name=args['train_policy_fn'], agents=agents
        )
        eval_policy_fn = get_policy_fn(
            name=args['eval_policy_fn'], agents=agents
        )

        train_rollout_worker = configure_rollout_worker(
            role='train', policy_fn=train_policy_fn, agents=agents, dims=input_dims,
            seed=rank_seed, logger=logger, params=args
        )
        eval_rollout_worker = configure_rollout_worker(
            role='eval', policy_fn=eval_policy_fn, agents=agents, dims=input_dims,
            seed=rank_seed, logger=logger, params=args
        )

        # Begin main training loop
        if rank == 0:
            start_time = time.time()

        if args['do_demo_only'] is False:
            training.train(
                memory=memory, agents=agents, saver=saver, sess=sess,
                train_rollout_worker=train_rollout_worker, eval_rollout_worker=eval_rollout_worker,
                param_noise_adaption_interval=50, **args
            )
        else:
            demo.demo(agents=agents, eval_rollout_worker=eval_rollout_worker,
                      demo_video_recording_name=args["demo_video_recording_name"])

        train_rollout_worker.close()
        eval_rollout_worker.close()

        if rank == 0:
            logger.info('total runtime: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, value in DEFAULT_PARAMS.items():
        key = '--' + key.replace('_', '-')
        parser.add_argument(key, type=get_convert_arg_to_type_fn(type(value)), default=value)

    args = parser.parse_args()
    dict_args = vars(args)

    logger.configure()

    dict_args = prepare_params(dict_args)
    log_params(dict_args)
    run(dict_args)

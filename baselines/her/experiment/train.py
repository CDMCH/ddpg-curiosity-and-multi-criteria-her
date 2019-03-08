import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork
from baselines.her.experiment.config import convert_arg_string_to_type

from subprocess import CalledProcessError

from collections import OrderedDict
import gym_boxpush
from gym_boxpush.heatmap_gen import generate_boxpush_heatmap_from_npy_records

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, rollout_batches_per_cycle, n_batches, policy_save_interval,
          save_policies, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1

    batch = 0

    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for cycle_index in range(n_cycles):
            for _ in range(rollout_batches_per_cycle):
                render_now = False
                # if rank == 0 and epoch >= 40:
                #     render_now=True
                episode = rollout_worker.generate_rollouts(
                    render_override=render_now,
                    heatmap_prefix='epoch{}_rank{}'.format(epoch, rank)
                )
                rollout_worker.envs[0].flush_record_write()
                policy.store_episode(episode)

            distances = policy.adapt_param_noises()

            for _ in range(n_batches):

                loss_dict = policy.train()

                logger.record_tabular('batch', batch)
                for key, val in loss_dict:
                    logger.record_tabular(key, mpi_average(val))
                if rank == 0:
                    logger.dump_tabular(format_types=logger.TensorBoardOutputFormat)
                batch += 1
            policy.update_target_net()

        generate_boxpush_heatmap_from_npy_records(
            read_dir=os.path.join(logger.get_dir(), 'heatmaps'),
            file_prefix='epoch{}'.format(epoch),
            delete_records=True
        )

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # Q_pis = policy.sess.run(policy.explore_networks.main.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape(
        #         [[10, 90, 0, 0],
        #          [90, 90, 0, 0],
        #          [10, 10, 0, 0],
        #          [90, 10, 0, 0],
        #          [40, 40, 0, 0]],
        #         (-1, policy.dimo)),
        #     policy.explore_networks.main.g_tf: np.reshape(
        #         [[50, 50],
        #          [50, 50],
        #          [50, 50],
        #          [50, 50],
        #          [50, 50]],
        #         (-1, policy.dimg))
        # })

        # actual_rewards = policy.sess.run(policy.reward_tensor_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape(
        #         [[10, 90, 0, 0],
        #          [90, 90, 0, 0],
        #          [10, 10, 0, 0],
        #          [90, 10, 0, 0],
        #          [40, 40, 0, 0]],
        #         (-1, policy.dimo)),
        #     policy.explore_networks.main.g_tf: np.reshape(
        #         [[50, 50],
        #          [50, 50],
        #          [50, 50],
        #          [50, 50],
        #          [50, 50]],
        #         (-1, policy.dimg))
        # })

        # target_tf = policy.sess.run(policy.explore_networks.target_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape(
        #         [[10, 90, 0, 0],
        #          [90, 90, 0, 0],
        #          [10, 10, 0, 0],
        #          [90, 10, 0, 0],
        #          [40, 40, 0, 0]],
        #         (-1, policy.dimo)),
        #     policy.explore_networks.main.g_tf: np.reshape(
        #         [[50, 50],
        #          [50, 50],
        #          [50, 50],
        #          [50, 50],
        #          [50, 50]],
        #         (-1, policy.dimg))
        # })

        # if rank == 0:
        #     print("Q_pis:\n{}\n".format(Q_pis))
            # print('actual_rewards:\n{}\n'.format(actual_rewards))
            # print("target_tf:\n{}\n".format(target_tf))


        #
        # test_q_vals['main_(90, 90)'] = policy.sess.run(policy.explore_networks.main.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([90, 90, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['main_(10, 10)'] = policy.sess.run(policy.explore_networks.main.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([10, 10, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['main_(90, 10)'] = policy.sess.run(policy.explore_networks.main.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([90, 10, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['main_(40, 40)'] = policy.sess.run(policy.explore_networks.main.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([40, 40, 0, 0], (-1, policy.dimo))
        # })[0]

        # test_q_vals['reward_(10, 90)'] = policy.sess.run(policy.reward_tensor_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([10, 90, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['reward_(90, 90)'] = policy.sess.run(policy.reward_tensor_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([90, 90, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['reward_(10, 10)'] = policy.sess.run(policy.reward_tensor_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([10, 10, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['reward_(90, 10)'] = policy.sess.run(policy.reward_tensor_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([90, 10, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['reward_(40, 40)'] = policy.sess.run(policy.reward_tensor_tf, feed_dict={
        #     policy.obs_input_for_reward_tf: np.reshape([40, 40, 0, 0], (-1, policy.dimo))
        # })[0]

        # test_q_vals['target_(10, 90)'] = policy.sess.run(policy.explore_networks.target.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward: np.reshape([10, 90, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['target_(90, 90)'] = policy.sess.run(policy.explore_networks.target.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward: np.reshape([90, 90, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['target_(10, 10)'] = policy.sess.run(policy.explore_networks.target.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward: np.reshape([10, 10, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['target_(90, 10)'] = policy.sess.run(policy.explore_networks.target.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward: np.reshape([90, 10, 0, 0], (-1, policy.dimo))
        # })[0]
        # test_q_vals['target_(40, 40)'] = policy.sess.run(policy.explore_networks.target.Q_pi_tf, feed_dict={
        #     policy.obs_input_for_reward: np.reshape([40, 40, 0, 0], (-1, policy.dimo))
        # })[0]

        # print('local: {}'.format(test_q_vals['main_(10, 90)']))
        # print('average: {}'.format(mpi_average(test_q_vals['main_(10, 90)'])))

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))
        # for key, val in test_q_vals.items():
        #     logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
    override_params={}, save_policies=True
):

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    # params.update(**override_params)  # makes it possible to override any parameter
    for k, v in override_params.items():
        if k in params:
            params[k] = convert_arg_string_to_type(arg_string=v, type=type(params[k]))
        else:
            raise ValueError('Unrecognized Parameter: {}'.format(k))

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps', 'intrinsic_motivation_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    render = False
    # if rank == 0:
        # render = True
        # rollout_params['noise_eps'] = 0.0
        # rollout_params['random_eps'] = 0.0
        # rollout_params['intrinsic_motivation_eps'] = 1.0

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, render=render, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'], rollout_batches_per_cycle=params['rollout_batches_per_cycle'],
        policy_save_interval=policy_save_interval, save_policies=save_policies)


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.pass_context
def main(ctx, **kwargs):

    override_params = dict()
    for item in ctx.args:
        override_params.update([item.split('=')])

    launch(override_params=override_params, **kwargs)


if __name__ == '__main__':
    main()

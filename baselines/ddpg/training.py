import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from baselines.common.mpi_moments import mpi_moments
import math
import time
from datetime import datetime
from pympler import asizeof
from baselines.ddpg.heatmap_gen import generate_3d_fetch_stack_heatmap_from_npy_records

# from gym_boxpush.heatmap_gen import generate_boxpush_heatmap_from_npy_records

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def train(memory, agents, saver, sess,
            train_rollout_worker, eval_rollout_worker, n_epochs, n_cycles, n_batches, batch_size, rollout_batches_per_cycle,
            n_test_rollouts, heatmaps, dynamics_loss_mapper, do_evaluation, save_at_score, stop_at_score, save_checkpoints_at, **kwargs):


    rank = MPI.COMM_WORLD.Get_rank()

    logger.info("Training...")
    best_success_rate = -1

    batch = 0

    should_quit_early = False

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = datetime.now()

        ### debug see how large a worker's replay buffer should be in memory
        # logger.info("\nreplay buffer self report sample size: {}".format(memory.nb_entries))
        # logger.info("replay buffer memory details:\n{}\n".format(asizeof.asized(memory, detail=1).format()))
        #
        # logger.info("logger memory details:\n{}\n".format(asizeof.asized(logger.Logger.CURRENT, detail=1).format()))
        ####

        if dynamics_loss_mapper is not None:
            dynamics_loss_mapper.set_record_write(prefix='epoch{}_rank{}'.format(epoch, rank))

        # train
        train_rollout_worker.clear_history()
        for cycle_index in range(n_cycles):
            for _ in range(rollout_batches_per_cycle):
                render_now = False
                # if rank == 0 and epoch >= 40:
                #     render_now=True

                episode = train_rollout_worker.generate_rollouts(
                    render_override=render_now,
                    heatmap_prefix='epoch{}_rank{}'.format(epoch, rank) if heatmaps else None
                )

                # TODO, make sure normalizers are updated properly
                memory.store_episode(episode)
                for agent in agents.values():
                    agent.update_normalizers(episode)

            param_noise_distances = {}

            # Adapt param noise.
            if memory.nb_entries >= batch_size:
                for role, agent in agents.items():
                    param_noise_distances[role] = agent.adapt_param_noise()

            for train_step in range(n_batches):
                critic_losses = {}
                actor_losses = {}
                for role, agent in agents.items():
                    # if MPI.COMM_WORLD.Get_rank() == 0:
                    #     print("training {}".format(role))
                    critic_losses[role], actor_losses[role] = agent.train()
                for agent in agents.values():
                    agent.update_target_net()

                # Commented out because it logs every single batch
                # logger.record_tabular('batch', batch)
                # for role in agents.keys():
                #     logger.record_tabular('{}_critic_loss'.format(role), mpi_average(critic_losses[role]))
                #     logger.record_tabular('{}_actor_loss'.format(role), mpi_average(actor_losses[role]))
                # if rank == 0:
                #     logger.dump_tabular(format_types=logger.TensorBoardOutputFormat)

                batch += 1

        if heatmaps:
            train_rollout_worker.flush_env_location_records()
            MPI.COMM_WORLD.Barrier()
            logger.info("Creating heatmap...")
            if rank == 0:
                heatmap_save_path = generate_3d_fetch_stack_heatmap_from_npy_records(
                    working_dir=os.path.join(logger.get_dir(), 'heatmaps'),
                    file_prefix='epoch{}'.format(epoch),
                    delete_records=True
                )
                logger.info("Heatmap saved to {}".format(heatmap_save_path))

        # Commented out to remove boxpush dependency
        #     train_rollout_worker.flush_record_writes()
        #
        #
        #     generate_boxpush_heatmap_from_npy_records(
        #         read_dir=os.path.join(logger.get_dir(), 'heatmaps'),
        #         write_dir=os.path.join(logger.get_dir(), 'heatmaps', 'rank_{}'.format(rank)),
        #         file_prefix='epoch{}_rank{}'.format(epoch, rank),
        #         delete_records=False
        #     )
        #
        #     if rank == 0:
        #         generate_boxpush_heatmap_from_npy_records(
        #             read_dir=os.path.join(logger.get_dir(), 'heatmaps'),
        #             file_prefix='epoch{}'.format(epoch),
        #             delete_records=True
        #         )
        #
        # if dynamics_loss_mapper is not None and rank == 0:
        #     dynamics_loss_mapper.flush_record_write()
        #     dynamics_loss_mapper.generate_dynamics_loss_map_from_npy_records(
        #         file_prefix='epoch{}'.format(epoch), delete_records=True
        #     )

        # test
        if do_evaluation:
            eval_rollout_worker.clear_history()
            for _ in range(n_test_rollouts):
                eval_rollout_worker.generate_rollouts()

            current_score = mpi_average(eval_rollout_worker.current_score())

            if current_score >= save_at_score and rank == 0:
                save_path = os.path.join(logger.get_dir(), 'saved_model', 'model.ckpt')
                logger.info("Saving models to {}".format(save_path))
                saver.save(sess, save_path)

            if save_checkpoints_at is not None:
                for score in save_checkpoints_at.copy():
                    if current_score >= score and rank == 0:
                        logger.info("Reached checkpoint for {}".format(score))
                        save_path = os.path.join(logger.get_dir(), 'saved_model', 'model_score_{}.ckpt'.format(str(score).replace(".", "p")))
                        logger.info("Saving models to {}".format(save_path))
                        saver.save(sess, save_path)
                        save_checkpoints_at.remove(score)

            if stop_at_score is not None and current_score >= stop_at_score:
                logger.info("Stopping score of {} reached. Quitting...".format(stop_at_score))
                should_quit_early = True

        # record logs
        logger.record_tabular('epoch', epoch)
        timesteps = MPI.COMM_WORLD.Get_size() * epoch * n_cycles * rollout_batches_per_cycle * train_rollout_worker.rollout_batch_size * train_rollout_worker.T
        logger.record_tabular('timesteps', timesteps)
        if do_evaluation:
            for key, val in eval_rollout_worker.logs('test'):
                logger.record_tabular(key, mpi_average(val))
        for key, val in train_rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for role, agent in agents.items():
            for key, val in agent.get_stats().items():
                logger.record_tabular("{}_agent_{}".format(role, key), mpi_average(val))
        # for key, val in test_q_vals.items():
        #     logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        # success_rate = mpi_average(eval_rollout_worker.current_success_rate())
        # if rank == 0 and success_rate >= best_success_rate and save_policies:
        #     best_success_rate = success_rate
        #     logger.info(
        #         'New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
        #     eval_rollout_worker.save_policy(best_policy_path)
        #     eval_rollout_worker.save_policy(latest_policy_path)
        # if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
        #     policy_path = periodic_policy_path.format(epoch)
        #     logger.info('Saving periodic policy to {} ...'.format(policy_path))
        #     eval_rollout_worker.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

        epoch_end_time = datetime.now()
        if rank == 0:
            logger.info("(epoch took {} seconds)".format((epoch_end_time-epoch_start_time).total_seconds()))
            logger.info("(completed at {})".format(epoch_end_time))

        if should_quit_early:
            break

    if rank == 0:
        save_path = os.path.join(logger.get_dir(), 'saved_model', 'model.ckpt')
        logger.info("Saving models to {}".format(save_path))
        saver.save(sess, save_path)

#TODO################################################################################################################
    # # Set up logging stuff only for a single worker.
    # if rank == 0:
    #     saver = tf.train.Saver()
    # else:
    #     saver = None
    #
    # step = 0
    # episode = 0
    # eval_episode_rewards_history = deque(maxlen=100)
    # episode_rewards_history = deque(maxlen=100)
    #
    # obs = env.reset()
    # if eval_env is not None:
    #     eval_obs = eval_env.reset()
    # done = False
    # episode_reward = 0.
    # episode_step = 0
    # episodes = 0
    # t = 0
    #
    # epoch = 0
    # start_time = time.time()
    #
    # epoch_episode_rewards = []
    # epoch_episode_steps = []
    # epoch_episode_eval_rewards = []
    # epoch_episode_eval_steps = []
    # epoch_start_time = time.time()
    # epoch_actions = []
    # epoch_qs = []
    # epoch_episodes = 0
    # for epoch in range(nb_epochs):
    #
    #     #todo
    #     write_dir = os.path.join(logger.get_dir(), 'heatmaps')
    #     env.unwrapped.set_record_write(write_dir=write_dir, prefix='epoch{}_rank{}'.format(epoch, rank))
    #
    #     for cycle in range(nb_epoch_cycles):
    #         # Perform rollouts.
    #         for t_rollout in range(nb_rollout_steps):
    #             # Predict next action.
    #             action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
    #             assert action.shape == env.action_space.shape
    #
    #             # Execute next action.
    #             if rank == 0 and (render or epoch >= 40):
    #                 env.render()
    #             assert max_action.shape == action.shape
    #             new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
    #             t += 1
    #             if rank == 0 and render:
    #                 env.render()
    #             episode_reward += r
    #             episode_step += 1
    #
    #             # Book-keeping.
    #             epoch_actions.append(action)
    #             epoch_qs.append(q)
    #             agent.store_transition(obs, action, r, new_obs, done)
    #             obs = new_obs
    #
    #             if done or t_rollout + 1 == nb_rollout_steps:
    #                 # Episode done.
    #                 epoch_episode_rewards.append(episode_reward)
    #                 episode_rewards_history.append(episode_reward)
    #                 epoch_episode_steps.append(episode_step)
    #                 episode_reward = 0.
    #                 episode_step = 0
    #                 epoch_episodes += 1
    #                 episodes += 1
    #
    #                 agent.reset()
    #                 obs = env.reset()
    #
    #         #todo
    #         env.unwrapped.flush_record_write()
    #
    #         # Train.
    #         epoch_actor_losses = []
    #         epoch_critic_losses = []
    #         epoch_adaptive_distances = []
    #         for t_train in range(nb_train_steps):
    #             # Adapt param noise, if necessary.
    #             if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
    #                 distance = agent.adapt_param_noise()
    #                 epoch_adaptive_distances.append(distance)
    #
    #             cl, al = agent.train()
    #             epoch_critic_losses.append(cl)
    #             epoch_actor_losses.append(al)
    #             agent.update_target_net()
    #
    #         # Evaluate.
    #         eval_episode_rewards = []
    #         eval_qs = []
    #         if eval_env is not None:
    #
    #             #todo
    #             env.unwrapped.log_location = False
    #
    #             eval_episode_reward = 0.
    #             for t_rollout in range(nb_eval_steps):
    #                 eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
    #                 eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
    #                 if render_eval:
    #                     eval_env.render()
    #                 eval_episode_reward += eval_r
    #
    #                 eval_qs.append(eval_q)
    #                 if eval_done:
    #                     eval_obs = eval_env.reset()
    #                     eval_episode_rewards.append(eval_episode_reward)
    #                     eval_episode_rewards_history.append(eval_episode_reward)
    #                     eval_episode_reward = 0.
    #
    #             #todo
    #             env.unwrapped.log_location = True
    #
    #     #todo
    #     generate_boxpush_heatmap_from_npy_records(
    #         directory=os.path.join(logger.get_dir(), 'heatmaps'),
    #         file_prefix='epoch{}'.format(epoch),
    #         delete_records=True
    #     )
    #
    #     mpi_size = MPI.COMM_WORLD.Get_size()
    #     # Log stats.
    #     # XXX shouldn't call np.mean on variable length lists
    #     duration = time.time() - start_time
    #     stats = agent.get_stats()
    #     combined_stats = stats.copy()
    #     combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
    #     combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
    #     combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
    #     combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
    #     combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
    #     combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
    #     combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
    #     combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
    #     combined_stats['total/duration'] = duration
    #     combined_stats['total/steps_per_second'] = float(t) / float(duration)
    #     combined_stats['total/episodes'] = episodes
    #     combined_stats['rollout/episodes'] = epoch_episodes
    #     combined_stats['rollout/actions_std'] = np.std(epoch_actions)
    #     # Evaluation statistics.
    #     if eval_env is not None:
    #         combined_stats['eval/return'] = eval_episode_rewards
    #         combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
    #         combined_stats['eval/Q'] = eval_qs
    #         combined_stats['eval/episodes'] = len(eval_episode_rewards)
    #     def as_scalar(x):
    #         if isinstance(x, np.ndarray):
    #             assert x.size == 1
    #             return x[0]
    #         elif np.isscalar(x):
    #             return x
    #         else:
    #             raise ValueError('expected scalar, got %s'%x)
    #     combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
    #     combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}
    #
    #     # Total statistics.
    #     combined_stats['total/epochs'] = epoch + 1
    #     combined_stats['total/steps'] = t
    #
    #     for key in sorted(combined_stats.keys()):
    #         logger.record_tabular(key, combined_stats[key])
    #     logger.dump_tabular()
    #     logger.info('')
    #     logdir = logger.get_dir()
    #     if rank == 0 and logdir:
    #         if hasattr(env, 'get_state'):
    #             with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
    #                 pickle.dump(env.get_state(), f)
    #         if eval_env and hasattr(eval_env, 'get_state'):
    #             with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
    #                 pickle.dump(eval_env.get_state(), f)

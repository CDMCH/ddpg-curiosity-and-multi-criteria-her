import os
from ddpg_curiosity_mc_her import logger
import numpy as np
from mpi4py import MPI
from ddpg_curiosity_mc_her.common.mpi_moments import mpi_moments
from datetime import datetime
from ddpg_curiosity_mc_her.ddpg.heatmap_gen import generate_3d_fetch_stack_heatmap_from_npy_records


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

    batch = 0

    should_quit_early = False

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = datetime.now()

        if dynamics_loss_mapper is not None:
            dynamics_loss_mapper.set_record_write(prefix='epoch{}_rank{}'.format(epoch, rank))

        # train
        train_rollout_worker.clear_history()
        for cycle_index in range(n_cycles):
            for _ in range(rollout_batches_per_cycle):

                episode = train_rollout_worker.generate_rollouts(
                    render_override=False,
                    heatmap_prefix='epoch{}_rank{}'.format(epoch, rank) if heatmaps else None
                )

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
                    critic_losses[role], actor_losses[role] = agent.train()
                for agent in agents.values():
                    agent.update_target_net()

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

        if rank == 0:
            logger.dump_tabular()

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

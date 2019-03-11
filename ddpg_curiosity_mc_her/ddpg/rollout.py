from collections import deque

import numpy as np
import pickle
import os
import cv2
import inspect
import functools

from ddpg_curiosity_mc_her import logger

try:
    from mujoco_py import MujocoException
except Exception:
    logger.warn("Mujoco could not be imported.")


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


class RolloutWorker:

    @store_args
    def __init__(self, policy_fn, agents, dims, logger, make_env, T, use_her, rollout_batch_size=1,
                 compute_Q=False, render=False, history_len=100):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]

        assert (np.abs(self.envs[0].action_space.low) == self.envs[0].action_space.high).all()  # we assume symmetric actions.
        self.max_action = self.envs[0].action_space.high
        logger.info('Scaling actions by {} before executing in env'.format(self.max_action))

        assert self.T > 0
        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        if self.use_her:
            self.success_history = deque(maxlen=history_len)
        self.reward_per_episode_history = deque(maxlen=history_len)

        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        if self.use_her:
            self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
            self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.total_reward_this_episode = np.zeros((self.rollout_batch_size,), np.float32)

        self.reset_all(force_env_resets=True)
        self.clear_history()

        self.current_heatmap_prefix = None

        self.recording = False

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        if self.use_her:
            self.initial_o[i] = obs['observation']
            self.initial_ag[i] = obs['achieved_goal']
            self.g[i] = obs['desired_goal']
        else:
            self.initial_o[i] = obs
        self.total_reward_this_episode[i] = 0.

    def reset_all(self, force_env_resets=False):
        """Resets all `rollout_batch_size` rollout workers.
        """

        if self.use_her or force_env_resets:
            for i in range(self.rollout_batch_size):
                self.reset_rollout(i)

        for agent in self.agents.values():
            agent.reset()

    def generate_rollouts(self, render_override=False, reset_on_success_overrride=False, heatmap_prefix=None,
                          demo_video_recording_name=None):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """

        if demo_video_recording_name is not None:
            if not self.recording:
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                self.out = cv2.VideoWriter('{}.avi'.format(demo_video_recording_name), fourcc, 24.0, (2560, 1440))
                self.recording = False


        if heatmap_prefix != self.current_heatmap_prefix:
            write_dir = os.path.join(logger.get_dir(), 'heatmaps')
            self.envs[0].unwrapped.set_location_record_name(write_dir=write_dir, prefix=heatmap_prefix)
            self.current_heatmap_prefix = heatmap_prefix

        self.reset_all(force_env_resets=False)

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        o[:] = self.initial_o

        if self.use_her:
            ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
            ag[:] = self.initial_ag

        # generate episodes
        obs, actions, rewards = [], [], []
        if self.use_her:
            achieved_goals, goals, successes = [], [], []
        else:
            dones = []

        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):

            policy_output = self.policy_fn(
                observation=o,
                goal=self.g if self.use_her else None,
                compute_Q=self.compute_Q
            )

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            rewards_new = np.empty((self.rollout_batch_size, 1))
            if self.use_her:
                ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
                success = np.zeros(self.rollout_batch_size)
            else:
                dones_new = np.empty((self.rollout_batch_size, 1))

            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    curr_o_new, curr_reward_new, curr_done_new, info = self.envs[i].step(u[i] * self.max_action)
                    # if shift_rewards_by_50:
                    #     curr_reward_new = curr_reward_new * 50 + 50
                        # curr_reward_new = curr_reward_new / 1000.
                    # else:
                    #     curr_reward_new = curr_reward_new * 50 + 50

                    rewards_new[i] = curr_reward_new
                    self.total_reward_this_episode[i] += curr_reward_new
                    if self.use_her:
                        if 'is_success' in info:
                            success[i] = info['is_success']
                            if reset_on_success_overrride and success[i]:
                                # raise Exception('SUCCESS affected rollout behavior')
                                # print("YOU SHOULD ONLY EVER REACH THIS IF CONDUCTING A DEMO")
                                self.reward_per_episode_history.append(self.total_reward_this_episode[i])
                                self.reset_rollout(i)

                        o_new[i] = curr_o_new['observation']
                        ag_new[i] = curr_o_new['achieved_goal']
                    else:
                        o_new[i] = curr_o_new
                        dones_new[i] = curr_done_new

                        if curr_done_new:
                            self.reward_per_episode_history.append(self.total_reward_this_episode[i])
                            self.reset_rollout(i)

                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]

                    if self.render or render_override:
                        if i == 0:
                            self.envs[0].render()
                            if demo_video_recording_name is not None:
                                frame = self.envs[i].render('rgb_array')[...,::-1]
                                cv2.imshow("recording {}".format(i), frame)
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('r'):
                                    if not self.recording:
                                        print("\n\n-------RECORDING---------\n\n")
                                        self.recording = True
                                if self.recording:
                                    print("rec {}".format(t))
                                    self.out.write(frame)
                                if key == ord('q'):
                                    self.out.release()
                                    cv2.destroyAllWindows()
                                    print('done')
                                    exit()

                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all(force_env_resets=True)
                return self.generate_rollouts()

            obs.append(o.copy())
            actions.append(u.copy())
            rewards.append(rewards_new.copy())
            o[...] = o_new
            if self.use_her:
                achieved_goals.append(ag.copy())
                successes.append(success.copy())
                goals.append(self.g.copy())
                ag[...] = ag_new
            else:
                dones.append(dones_new.copy())

        obs.append(o.copy())
        self.initial_o[:] = o

        if self.use_her:
            achieved_goals.append(ag.copy())

        episode = {'o': obs, 'u': actions}
        episode['r'] = rewards

        if self.use_her:
            episode['g'] = goals
            episode['ag'] = achieved_goals
        else:
            episode['t'] = dones

        # print("goals shape: {}".format(np.shape( episode['g'])))
        # print("obs shape: {}".format(np.shape(episode['o'])))
        # print("ag shape: {}".format(np.shape(episode['ag'])))

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        if self.use_her:
            successful = np.array(successes)[-1, :]
            assert successful.shape == (self.rollout_batch_size,)
            success_rate = np.mean(successful)
            self.success_history.append(success_rate)
            self.reward_per_episode_history.append(self.total_reward_this_episode[i])
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        # print("goals shape: {}".format(np.shape( episode['g'])))
        # print("obs shape: {}".format(np.shape(episode['o'])))
        # print("ag shape: {}".format(np.shape(episode['ag'])))

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        if self.use_her:
            self.success_history.clear()
        self.reward_per_episode_history.clear()
        self.Q_history.clear()

    def flush_env_location_records(self):
        self.envs[0].unwrapped.flush_location_record()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_reward_per_episode(self):
        return np.mean(self.reward_per_episode_history)

    def current_score(self):
        if self.use_her:
            return self.current_success_rate()
        else:
            return self.current_mean_reward_per_episode()

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.agents, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        if self.use_her:
            logs += [('success_rate', np.mean(self.success_history))]
        logs += [('reward_per_episode', np.mean(self.reward_per_episode_history))]

        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)

    def close(self):
        for env in self.envs:
            env.close()

from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from mpi4py import MPI

from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer
from baselines.her.ddpg_network_pair import DDPGNetworkPair
from baselines.her.forward_dynamic import ForwardDynamics

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

def reset(self):
    # Reset internal state after an episode is complete.
    if self.action_noise is not None:
        self.action_noise.reset()
    if self.param_noise is not None:
        self.sess.run(self.perturb_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., intrinsic_motivation_eps=0., use_target_net=False,
                    compute_Q=False, apply_param_noise_if_avail=False):
        o, g = self._preprocess_og(o, ag, g)

        networks = [self.exploit_networks, self.explore_networks][np.random.binomial(1, intrinsic_motivation_eps)]

        if intrinsic_motivation_eps == 0:
            assert networks == self.exploit_networks

        policy = networks.target if use_target_net else networks.main
        # values to compute
        if policy.use_param_noise and apply_param_noise_if_avail:
            vals = [policy.param_noise_pi_tf]
        else:
            vals = [policy.pi_tf]

        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        # self.exploit_networks.Q_adam.sync()
        # self.exploit_networks.pi_adam.sync()
        for optimizer in self.optimizer_by_grad_key_tf.values():
            optimizer.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        run_results_list = self.sess.run(list(self.train_run_vals_tf.values()))
        run_results_dict = OrderedDict(zip(self.train_run_vals_tf.keys(), run_results_list))
        return run_results_dict

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def adapt_param_noises(self):
        if self.use_param_noise is None:
            return 0.

        batch = None

        network_pairs = [self.exploit_networks, self.explore_networks]
        mean_distances = []

        for network_pair in network_pairs:
            param_noise_spec = network_pair.param_noise_spec

            if param_noise_spec is None:
                mean_distance = 0
            else:
                perturb_adaptive_policy_ops = network_pair.main.perturb_adaptive_policy_ops
                adaptive_policy_distance = network_pair.main.adaptive_policy_distance
                param_noise_stddev_tf = network_pair.main.param_noise_stddev_tf

                if batch is None:
                    batch = self.sample_batch()
                self.stage_batch(batch)

                # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
                self.sess.run(perturb_adaptive_policy_ops, feed_dict={
                    param_noise_stddev_tf: param_noise_spec.current_stddev,
                })
                distance = self.sess.run(adaptive_policy_distance)

                mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
                param_noise_spec.adapt(mean_distance)

            mean_distances.append(mean_distance)

        return mean_distances

    def reset_param_noises(self):
        network_pairs = [self.exploit_networks, self.explore_networks]
        for network_pair in network_pairs:
            param_noise_spec = network_pair.param_noise_spec
            if param_noise_spec is not None:
                perturb_policy_ops = network_pair.main.perturb_policy_ops
                param_noise_stddev_tf = network_pair.main.param_noise_stddev_tf

                self.sess.run(perturb_policy_ops, feed_dict={
                    param_noise_stddev_tf: param_noise_spec.current_stddev,
                })

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        run_results = self._grads()

        for grad_key in self.optimizer_by_grad_key_tf.keys():
            grad = run_results[grad_key]
            optimizer = self.optimizer_by_grad_key_tf[grad_key]
            optimizer.update(grad, self.Q_lr)

        return [(loss_key, run_results[loss_key]) for loss_key in self.loss_keys]

    def _init_target_net(self):
        self.sess.run([
            self.exploit_networks.init_target_net_op,
            self.explore_networks.init_target_net_op
        ])

    def update_target_net(self):
        self.sess.run([
            self.exploit_networks.update_target_net_op,
            self.explore_networks.update_target_net_op
        ])

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        with tf.variable_scope('exploit_networks'):
            self.exploit_networks = DDPGNetworkPair(batch_tf=batch_tf, reuse=reuse, input_goals=True, create_actor_critic_fn=self.create_actor_critic,
                                              polyak_coeffient=self.polyak, clip_return=self.clip_return, clip_pos_returns=self.clip_pos_returns,
                                              gamma=self.gamma, dimu=self.dimu, max_u=self.max_u, action_l2=self.action_l2, o_stats=self.o_stats,
                                                    g_stats=self.g_stats, ac_hidden=self.hidden, ac_layers=self.layers, use_layer_norm=self.use_layer_norm, use_param_noise=self.use_param_noise, param_noise_stddev=self.param_noise_stddev)

        with tf.variable_scope('dynamics_model'):
            self.dynamics_model = ForwardDynamics(inputs_tf=batch_tf,
                                                  dimo=self.dimo,
                                                  dimu=self.dimu,
                                                  max_u=self.max_u,
                                                  o_stats=self.o_stats,
                                                  hidden=self.hidden,
                                                  layers=self.layers
                                                  )

        # print(self.dimg)
        # print(self.dynamics_model.per_sample_loss_tf.shape)
        # print(self.dynamics_model.per_sample_loss_tf.shape[0])
        # exit(0)

        # with tf.variable_scope('dynamics_loss_stats') as vs:
        #     if reuse:
        #         vs.reuse_variables()
        #     self.dynamics_loss_stats = Normalizer(1, self.norm_eps, self.norm_clip, sess=self.sess, new_sample_weight=1.001)

        explore_batch_tf = batch_tf.copy()
        self.original_reward_tf = explore_batch_tf['r']

        # print("per sample loss shape: {}".format(self.dynamics_model.per_sample_loss_tf.shape))
        # print("original reward shape: {}".format(explore_batch_tf['r'].shape))
        # assert explore_batch_tf['r'].shape == self.dynamics_model.per_sample_loss_tf.shape

        explore_batch_tf['r'] = tf.clip_by_value(tf.stop_gradient(self.dynamics_model.per_sample_loss_tf), -100, 100)

        # def is_player_in_area(single_obs):
        #     assert single_obs.shape == [4]
        #     condition = tf.logical_and(
        #         tf.less(single_obs[0], 30),
        #         tf.greater(single_obs[1], 70))
        #     return tf.cond(condition,
        #      true_fn=lambda: tf.constant(1, dtype=tf.float32),
        #      false_fn=lambda: tf.constant(-1, dtype=tf.float32),
        #
        #  )
        # explore_batch_tf['r'] = tf.stop_gradient(tf.expand_dims(tf.map_fn(is_player_in_area, explore_batch_tf['o']),axis=1))
        # # explore_batch_tf['r'] = tf.Print(explore_batch_tf['r'], data=[explore_batch_tf['r'], tf.shape(explore_batch_tf['r']), explore_batch_tf['o'], tf.shape(explore_batch_tf['o']), self.original_reward_tf, tf.shape(self.original_reward_tf)], message='rewards and observations, and original reward')
        # self.reward_tensor_tf = explore_batch_tf['r']
        # self.obs_input_for_reward_tf = explore_batch_tf['o']

        with tf.variable_scope('explore_networks'):
            self.explore_networks = DDPGNetworkPair(batch_tf=explore_batch_tf, reuse=reuse, input_goals=False, create_actor_critic_fn=self.create_actor_critic,
                                              polyak_coeffient=self.polyak, clip_return=self.clip_return, clip_pos_returns=self.clip_pos_returns,
                                              gamma=self.gamma, dimu=self.dimu, max_u=self.max_u, action_l2=self.action_l2, o_stats=self.o_stats,
                                                    g_stats=self.g_stats, ac_hidden=self.hidden, ac_layers=self.layers, use_layer_norm=self.use_layer_norm, use_param_noise=self.use_param_noise, param_noise_stddev=self.param_noise_stddev)


        networks = OrderedDict([
            ('exploit', self.exploit_networks),
            ('dynamics', self.dynamics_model),
            ('explore', self.explore_networks)
        ])

        self.train_run_vals_tf = OrderedDict()
        self.optimizer_by_grad_key_tf = OrderedDict()
        self.loss_keys = []

        for network_key in networks.keys():
            for run_value_key in networks[network_key].train_run_vals_tf.keys():
                run_value = networks[network_key].train_run_vals_tf[run_value_key]
                combined_key = '{}_{}'.format(network_key, run_value_key)
                self.train_run_vals_tf[combined_key] = run_value

                if run_value_key.endswith('_grad'):
                    optimizer = networks[network_key].optimizers_by_grad_key_tf[run_value_key]
                    self.optimizer_by_grad_key_tf[combined_key] = optimizer

                if run_value_key.endswith('_loss'):
                    self.loss_keys.append(combined_key)

        # print("self.train_run_vals_tf: {}".format(self.train_run_vals_tf))
        #
        # print("first 4 {}".format(list(self.train_run_vals_tf.values())[:4]))
        #
        # exit(0)

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        # logs += [('stats_dynamics_loss/mean', np.mean(self.sess.run([self.dynamics_loss_stats.mean])))]
        # logs += [('stats_dynamics_loss/std', np.mean(self.sess.run([self.dynamics_loss_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        #TODO: update exxluded subnames
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

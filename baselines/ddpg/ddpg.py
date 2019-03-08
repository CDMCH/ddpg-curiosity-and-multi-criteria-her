from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines.ddpg.dynamics import ForwardDynamics, RandomNetworkDistillation
from baselines.ddpg.models import actor, critic

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.ddpg.PopArtMeanStd import PopArtMeanStd
from mpi4py import MPI


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def _vars(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def _trainable_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def _perturbable_vars(scope):
    return [var for var in _trainable_vars(scope) if 'LayerNorm' not in var.name]


def _output_vars(scope):
    output_vars = [var for var in _vars(scope) if 'output' in var.name]
    return output_vars

def _vars_except_output(scope):
    relevant_vars = [var for var in _vars(scope) if 'output' not in var.name]
    return relevant_vars

def _trainable_vars_except_output(scope):
    relevant_vars = [var for var in _trainable_vars(scope) if 'output' not in var.name]
    return relevant_vars


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor_scope, perturbed_actor_scope, param_noise_stddev):
    actor_vars = _vars(actor_scope)
    perturbable_actor_vars = _perturbable_vars(actor_scope)
    perturbed_actor_vars = _vars(perturbed_actor_scope)
    perturbable_perturbed_actor_vars = _perturbable_vars(perturbed_actor_scope)

    assert len(actor_vars) == len(perturbed_actor_vars)
    assert len(perturbable_actor_vars) == len(perturbable_perturbed_actor_vars)

    updates = []
    for var, perturbed_var in zip(actor_vars, perturbed_actor_vars):
        if var in perturbable_actor_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor_vars)
    return tf.group(*updates)


class DDPG(object):
    def __init__(self, sess, scope, layer_norm, nb_actions, memory, use_intrinsic_reward, use_goals, observation_shape, action_shape, agent_hidden_layer_sizes, comm,
                 goal_shape=None, param_noise=None, action_noise=None, dynamics_hidden=None, dynamics_layers=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True, normalize_goals=True,
        batch_size=128, observation_range=(-5., 5.), goal_range=(-200, 200), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., dynamics_normalize_observations=False,
                 dynamics_loss_mapper=None, mix_external_critic_with_internal=None, external_critic_fn=None, intrinsic_motivation_method='forward_dynamics'):

        with tf.variable_scope(scope) as vs:

            # Inputs.
            self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
            self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
            if use_goals:
                self.goal0 = tf.placeholder(tf.float32, shape=(None,) + goal_shape, name='goal0')
                self.goal1 = tf.placeholder(tf.float32, shape=(None,) + goal_shape, name='goal1')
            else:
                self.goal0 = None
                self.goal1 = None
                if not use_intrinsic_reward:
                    self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
            self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
            self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
            self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
            self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

            # Parameters.
            self.scope = vs
            self.nb_actions = nb_actions
            self.layer_norm = layer_norm
            self.gamma = gamma
            self.tau = tau
            self.memory = memory
            self.normalize_observations = normalize_observations
            self.normalize_goals = normalize_goals
            self.normalize_returns = normalize_returns
            self.action_noise = action_noise
            self.param_noise = param_noise
            self.action_shape = action_shape
            self.action_range = action_range
            self.return_range = return_range
            self.observation_range = observation_range
            self.goal_range = goal_range
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr
            self.clip_norm = clip_norm
            self.enable_popart = enable_popart
            self.reward_scale = reward_scale
            self.batch_size = batch_size
            self.stats_sample = None
            self.critic_l2_reg = critic_l2_reg
            self.observation_shape = observation_shape
            self.goal_shape = goal_shape
            self.use_goals = use_goals
            self.use_intrinsic_reward = use_intrinsic_reward
            self.agent_hidden_layer_sizes = agent_hidden_layer_sizes
            self.dynamics_loss_mapper = dynamics_loss_mapper
            self.sess = sess
            self.mix_external_critic_with_internal = mix_external_critic_with_internal
            self.comm = comm

            assert not (self.mix_external_critic_with_internal is not None and external_critic_fn is None)
            assert not (self.use_intrinsic_reward and self.use_goals and mix_external_critic_with_internal is None)

            # Observation normalization.
            if self.normalize_observations:
                with tf.variable_scope('obs_rms'):
                    self.obs_rms = RunningMeanStd(shape=observation_shape)
            else:
                self.obs_rms = None

            normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
                self.observation_range[0], self.observation_range[1])
            normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
                self.observation_range[0], self.observation_range[1])

            # Goal normalization.
            if self.use_goals and self.normalize_goals and goal_shape:
                with tf.variable_scope('goal_rms'):
                    self.goal_rms = RunningMeanStd(shape=goal_shape)
            else:
                self.goal_rms = None
            if use_goals:
                normalized_goal0 = tf.clip_by_value(normalize(self.goal0, self.goal_rms),
                    self.goal_range[0], self.goal_range[1])
                normalized_goal1 = tf.clip_by_value(normalize(self.goal1, self.goal_rms),
                    self.goal_range[0], self.goal_range[1])
            else:
                normalized_goal0 = None
                normalized_goal1 = None

            # Create target networks.
            # target_actor = copy(actor)
            # target_actor.name = 'target_actor'
            # self.target_actor = target_actor
            # target_critic = copy(critic)
            # target_critic.name = 'target_critic'
            # self.target_critic = target_critic

            # Create networks and core TF parts that are shared across setup parts.
            self.actor_tf, self.actor_scope, self.actor_preactivations_tf = actor(obs=normalized_obs0, nb_actions=self.nb_actions, use_goals=self.use_goals, goal=normalized_goal0, name='actor', layer_norm=self.layer_norm, reuse=False, hidden_layer_sizes=agent_hidden_layer_sizes, also_return_preactivations=True)
            
            if self.mix_external_critic_with_internal is not None:
                self.combined_actor_tf, self.combined_actor_scope, self.combined_actor_preactivations_tf = actor(obs=normalized_obs0, nb_actions=self.nb_actions, use_goals=self.use_goals, goal=normalized_goal0, name='combined_actor', layer_norm=self.layer_norm, reuse=False, hidden_layer_sizes=agent_hidden_layer_sizes, also_return_preactivations=True)
            
            self.normalized_critic_tf, self.critic_scope, _ = critic(obs=normalized_obs0, action=self.actions, use_goals=self.use_goals, goal=normalized_goal0, name='critic', layer_norm=self.layer_norm, reuse=False, hidden_layer_sizes=agent_hidden_layer_sizes)
            self.normalized_critic_with_actor_tf, _, self.critic_with_actor_before_pop_tf = critic(obs=normalized_obs0, action=self.actor_tf, use_goals=self.use_goals, goal=normalized_goal0, name='critic', layer_norm=self.layer_norm, reuse=True, hidden_layer_sizes=agent_hidden_layer_sizes)
            if self.mix_external_critic_with_internal is not None:
                self.normalized_critic_with_combined_actor_tf, _, self.critic_with_combined_actor_before_pop_tf = critic(obs=normalized_obs0, action=self.combined_actor_tf, use_goals=self.use_goals, goal=normalized_goal0, name='critic', layer_norm=self.layer_norm, reuse=True, hidden_layer_sizes=agent_hidden_layer_sizes)


            def critic_with_actor_fn(actor_tf, normalized_obs0, normalized_goal0):
                with tf.variable_scope(scope):
                    assert scope == 'exploit_ddpg'
                    output = critic(obs=normalized_obs0, action=actor_tf, use_goals=self.use_goals, goal=normalized_goal0,
                           name='critic', layer_norm=self.layer_norm, reuse=True,
                           hidden_layer_sizes=agent_hidden_layer_sizes)
                return output
            self.critic_with_actor_fn = critic_with_actor_fn

        if self.mix_external_critic_with_internal:
            self.normalized_external_critic_with_combined_actor_tf, _, _ = external_critic_fn(self.combined_actor_tf, normalized_obs0, normalized_goal0)

        with tf.variable_scope(scope) as vs:

            target_actor_tf, self.target_actor_scope = actor(obs=normalized_obs1, nb_actions=self.nb_actions, use_goals=self.use_goals, goal=normalized_goal1, name='target_actor', layer_norm=self.layer_norm, reuse=False, hidden_layer_sizes=agent_hidden_layer_sizes)
            normalized_target_critic_tf, self.target_critic_scope, _ = critic(obs=normalized_obs1, action=target_actor_tf, use_goals=self.use_goals, goal=normalized_goal1, name='target_critic', layer_norm=self.layer_norm, reuse=False, hidden_layer_sizes=agent_hidden_layer_sizes)

            self.actor_vars = _vars(self.actor_scope)
            self.actor_trainable_vars = _trainable_vars(self.actor_scope)
            if self.mix_external_critic_with_internal is not None:
                self.combined_actor_vars = _vars(self.combined_actor_scope)
                self.combined_actor_trainable_vars = _trainable_vars(self.combined_actor_scope)

            self.critic_vars = _vars(self.critic_scope)
            self.critic_trainable_vars = _trainable_vars(self.critic_scope)
            self.critic_output_vars = _output_vars(self.critic_scope)

            self.target_actor_vars = _vars(self.target_actor_scope)
            self.target_critic_vars = _vars(self.target_critic_scope)
            self.target_critic_output_vars = _output_vars(self.target_critic_scope)

            self.critic_vars_except_output = _vars_except_output(self.critic_scope)
            self.target_critic_vars_except_output = _vars_except_output(self.target_critic_scope)

            if self.normalize_returns and self.enable_popart:
                self.setup_popart()
            elif self.normalize_returns:
                self.ret_rms = None
                assert False
            else:
                self.ret_rms = None

            self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
            self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
            if self.mix_external_critic_with_internal is not None:
                self.critic_with_combined_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_combined_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

            Q_obs1 = denormalize(normalized_target_critic_tf, self.ret_rms)

            # Don't input terminals when working as a Universal Value Function Approximator or maximizing dynamics loss
            if self.use_goals or self.use_intrinsic_reward:
                self.target_Q = self.rewards + gamma * Q_obs1
            else:
                self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

            # TODO: this can be removed after debug is over:
            # self.pop_hasnt_been_active = True
            # self.get_critic_var_values = U.GetFlat(self.critic_trainable_vars)
            # self.get_actor_var_values = U.GetFlat(self.actor_trainable_vars)

            ####

            # Set up dynamics
            if self.use_intrinsic_reward:
                if dynamics_normalize_observations:
                    logger.info("Normalizing Dynamics Observations")
                    dynamics_obs_0_input = normalized_obs0
                    dynamics_obs_1_input = normalized_obs1
                else:
                    logger.info("Dynamics Observations are not Normalized")
                    dynamics_obs_0_input = self.obs0
                    dynamics_obs_1_input = self.obs1

                if intrinsic_motivation_method == 'forward_dynamics':
                    self.dynamics = ForwardDynamics(
                        dynamics_obs_0_input, self.actions, dynamics_obs_1_input, self.clip_norm, hidden=dynamics_hidden,
                        layers=dynamics_layers, comm=comm
                    )
                elif intrinsic_motivation_method == 'random_network_distillation':
                    print("possibly not fully implemented, check through code")
                    assert False
                    self.dynamics = RandomNetworkDistillation(
                        dynamics_obs_0_input, self.actions, dynamics_obs_1_input, self.clip_norm, hidden=dynamics_hidden,
                        layers=dynamics_layers
                    )
                else:
                    raise ValueError("intrinsic motivation method: {} not recognized".format(intrinsic_motivation_method))

            # Set up parts.
            if self.param_noise is not None:

                if self.mix_external_critic_with_internal is not None:
                    actor_scope_for_param_noise = self.combined_actor_scope
                    actor_tf_for_param_noise = self.combined_actor_tf
                else:
                    actor_scope_for_param_noise = self.actor_scope
                    actor_tf_for_param_noise = self.actor_tf

                self.setup_param_noise(actor_scope_for_param_noise, actor_tf_for_param_noise, normalized_obs0, normalized_goal0)
            self.setup_actor_optimizer()
            self.setup_critic_optimizer()

            self.setup_stats()
            self.setup_target_network_updates()

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor_vars, self.target_actor_vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic_vars_except_output, self.target_critic_vars_except_output, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, actor_scope, actor_tf, normalized_obs0, normalized_goal0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        # param_noise_actor = copy(self.actor)
        # param_noise_actor.name = 'param_noise_actor'
        # self.perturbed_actor_tf = param_noise_actor(normalized_obs0, normalized_goal0)

        self.perturbed_actor_tf, self.perturbed_actor_scope = actor(obs=normalized_obs0, nb_actions=self.nb_actions, use_goals=self.use_goals, goal=normalized_goal0, name='param_noise_actor', layer_norm=self.layer_norm, reuse=False, hidden_layer_sizes=self.agent_hidden_layer_sizes)

        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(actor_scope, self.perturbed_actor_scope, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        # adaptive_param_noise_actor = copy(self.actor)
        # adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        # adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0, normalized_goal0)

        adaptive_actor_tf, adaptive_actor_scope = actor(obs=normalized_obs0, nb_actions=self.nb_actions, use_goals=self.use_goals, goal=normalized_goal0,
              name='adaptive_param_noise_actor', layer_norm=self.layer_norm, reuse=False, hidden_layer_sizes=self.agent_hidden_layer_sizes)

        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(actor_scope, adaptive_actor_scope, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')

        action_l2_coefficient = 0.001

        if self.mix_external_critic_with_internal is not None:
            external_weight = self.mix_external_critic_with_internal[0]
            internal_weight = self.mix_external_critic_with_internal[1]
            self.combined_actor_loss = (-tf.reduce_mean(self.normalized_critic_with_combined_actor_tf) * internal_weight)
            self.combined_actor_loss += (-tf.reduce_mean(self.normalized_external_critic_with_combined_actor_tf) * external_weight)
            self.combined_actor_loss += action_l2_coefficient * tf.reduce_mean(tf.square(self.combined_actor_preactivations_tf))

            combined_actor_shapes = [var.get_shape().as_list() for var in self.combined_actor_trainable_vars]
            combined_actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in combined_actor_shapes])
            logger.info('  combined_actor shapes: {}'.format(combined_actor_shapes))
            logger.info('  combined_actor params: {}'.format(combined_actor_nb_params))

            assert len(self.combined_actor_trainable_vars) > 0
            self.combined_actor_grads = U.flatgrad(self.combined_actor_loss, self.combined_actor_trainable_vars, clip_norm=self.clip_norm)
            self.combined_actor_optimizer = MpiAdam(var_list=self.combined_actor_trainable_vars,
                                                    beta1=0.9, beta2=0.999, epsilon=1e-08, comm=self.comm)

            # self.actor_loss = -tf.reduce_mean(self.critic_with_actor_before_pop_tf)
        self.actor_loss = -tf.reduce_mean(self.normalized_critic_with_actor_tf)
        self.actor_loss += action_l2_coefficient * tf.reduce_mean(tf.square(self.actor_preactivations_tf))

        actor_shapes = [var.get_shape().as_list() for var in self.actor_trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))

        assert len(self.actor_trainable_vars) > 0
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor_trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor_trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08, comm=self.comm)


    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')


        self.normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - self.normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic_trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic_trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))

        assert len(self.critic_trainable_vars) > 0
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic_trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic_trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08, comm=self.comm)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        # self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        # new_std = self.ret_rms.std
        # self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        # new_mean = self.ret_rms.mean

        self.kernels = []
        self.biases = []

        # self.renormalize_Q_outputs_op = []
        for vs in [self.critic_output_vars, self.target_critic_output_vars]:
            # print("len vs: {}".format(len(vs)))
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            # self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            # self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

            self.kernels.append(M)
            self.biases.append(b)

        # Return normalization.
        with tf.variable_scope('ret_rms'):
            self.ret_rms = PopArtMeanStd(sess=self.sess, kernels=self.kernels, biases=self.biases, comm=self.comm)

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        if self.normalize_goals and self.use_goals:
            ops += [tf.reduce_mean(self.goal_rms.mean), tf.reduce_mean(self.goal_rms.std)]
            names += ['goal_rms_mean', 'goal_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def pi(self, obs, goal=None, apply_noise=True, compute_Q=True):
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            if self.mix_external_critic_with_internal is not None:
                actor_tf = self.combined_actor_tf
            else:
                actor_tf = self.actor_tf

        if self.mix_external_critic_with_internal is not None:
            critic_with_actor_tf = self.critic_with_combined_actor_tf
        else:
            critic_with_actor_tf = self.critic_with_actor_tf

        feed_dict = {self.obs0: np.reshape(obs, newshape=[-1, *self.observation_shape])}
        if self.use_goals:
            feed_dict[self.goal0] = np.reshape(goal, newshape=[-1, *self.goal_shape])

        if compute_Q:
            action, q = self.sess.run([actor_tf, critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
        # action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise(size=action.shape)
            # print('noise.shape: {}'.format(noise.shape))
            # print('action.shape: {}'.format(action.shape))
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])

        if compute_Q:
            return action, q
        else:
            return action

    # def store_transition(self, obs0, action, reward, obs1, terminal1):
    #     reward *= self.reward_scale
    #     self.memory.append(obs0, action, reward, obs1, terminal1)
    #     if self.normalize_observations:
    #         self.obs_rms.update(np.array([obs0]))

    def update_normalizers(self, episode_batch):
        if self.normalize_observations:
            self.obs_rms.update(np.reshape(episode_batch['o'], newshape=(-1, *self.observation_shape)))

        if self.use_goals and self.normalize_goals:
            episode_batch['o_1'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_1'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.memory.sample_transitions(episode_batch, num_normalizing_transitions)

            self.goal_rms.update(np.reshape(transitions['g'], newshape=(-1, *self.goal_shape)))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.use_intrinsic_reward:

            dynamics_grads, dynamics_loss, dynamics_per_sample_loss = self.sess.run(
                [self.dynamics.dynamics_grads, self.dynamics.mean_loss_tf, self.dynamics.per_sample_loss_tf],
                feed_dict={
                    self.obs0: batch['o'],
                    self.actions: batch['u'],
                    self.obs1: batch['o_1']
                })
            self.dynamics.dynamics_adam.update(dynamics_grads, stepsize=self.actor_lr)

            if self.dynamics_loss_mapper is not None:
                self.dynamics_loss_mapper.log_losses_at_locations(
                    losses=dynamics_per_sample_loss,
                    locations=batch['o'][:, :2]
                )

            # Sanity check that our intrinsic reward is the same shape as the actual reward:
            assert np.array_equal(batch['r'].shape, dynamics_per_sample_loss.shape)

            batch['r'] = dynamics_per_sample_loss * 1000

            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.record_tabular(self.scope.name + '_dynamics_loss', np.squeeze(dynamics_loss))

            #TODO remove this
            # goal = [0.7, 0.7]
            # distance = - np.sum(np.square(batch['o'][:, :2] - goal), axis=1).reshape([len(batch['o']), -1]) + 1
            #
            # if self.dynamics_loss_mapper is not None:
            #     self.dynamics_loss_mapper.log_losses_at_locations(
            #         losses=distance,
            #         locations=batch['o'][:, :2]
            #     )
            # # if MPI.COMM_WORLD.Get_rank() == 0:
            # #     print("distance min: {} max: {}".format(np.min(distance, axis=(0,1)), np.max(distance, axis=(0,1))))
            # #     print("\nobs:\n{}\n".format(batch['o']))
            # #     print("\nformatted_obs:\n{}\n".format(batch['o'][:, :2]))
            # #     print("\ndistance:\n{}\n".format(distance))
            # assert np.array_equal(batch['r'].shape, distance.shape)
            # batch['r'] = distance

        assert not np.isnan(np.sum(batch['r']))

        if self.normalize_returns and self.enable_popart:

            # TODO: Commented below is for tracking POP values
            # main_critic_kernel = self.sess.run(self.kernels[0])
            # target_critic_kernel = self.sess.run(self.kernels[1])
            # main_critic_bais = self.sess.run(self.biases[0])
            # target_critic_bais = self.sess.run(self.biases[1])
            #
            # diagonal = np.diagonal(main_critic_kernel)
            #
            # # Sanity checks
            # # if np.alltrue(main_critic_kernel == target_critic_kernel):
            # #     # if MPI.COMM_WORLD.Get_rank() == 0:
            # #     #     print('good')
            # #     pass
            # # else:
            # #     assert np.alltrue(main_critic_kernel == target_critic_kernel)
            # # assert np.alltrue(main_critic_bais == main_critic_bais)
            # # assert np.count_nonzero(main_critic_kernel - np.diag(np.diagonal(main_critic_kernel))) or len(np.diag(np.diagonal(main_critic_kernel))) == 1
            # # assert np.all([x == diagonal[0] for x in diagonal])
            #
            # if MPI.COMM_WORLD.Get_rank() == 0:
            #     logger.record_tabular(self.scope.name + '_critic_output_kernal_val', diagonal[0])
            #     logger.record_tabular(self.scope.name + '_critic_output_bias_val', np.squeeze(main_critic_bais))
            ####

            feed_dict = {
                self.obs1: batch['o_1'],
                self.rewards: batch['r'],
            }
            if self.use_goals:
                feed_dict[self.goal1] = batch['g']
            elif not self.use_intrinsic_reward:
                feed_dict[self.terminals1] = batch['t'].astype('float32')

            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q],
                                                        feed_dict=feed_dict)
            pop_is_active, new_mean, new_std = self.ret_rms.pop_art_update(target_Q.flatten())

            # TODO: This can exist without other debugging snippets, but it logs a datapoint every single batch. making tensorboard super slow
            # if MPI.COMM_WORLD.Get_rank() == 0:
            #     logger.record_tabular(self.scope.name + '_target_Q_mean_per_batch', np.squeeze(np.mean(target_Q)))
            #     logger.record_tabular(self.scope.name + '_avg_reward_per_batch', np.squeeze(np.mean(batch['r'])))
            #     logger.record_tabular(self.scope.name + '_pop_is_active', np.squeeze(pop_is_active))
            #     logger.record_tabular(self.scope.name + '_new_ret_std_per_batch', np.squeeze(new_std))
            #     logger.record_tabular(self.scope.name + '_new_ret_mean_per_batch', np.squeeze(new_mean))
            #     logger.record_tabular(self.scope.name + '_obs_mean_per_batch', np.squeeze(np.mean(batch['o'])))
            #     logger.record_tabular(self.scope.name + '_obs_std_per_batch', np.squeeze(np.std(batch['o'])))
            #######


            #
            # self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
            #     self.old_std : np.array([old_std]),
            #     self.old_mean : np.array([old_mean]),
            # })

            # Run sanity check. Disabled by default since it slows down things considerably.
            # print('running sanity check')

            # TODO: Commented below are all POPART sanity checks
            # new_main_critic_kernel = self.sess.run(self.kernels[0])
            # new_target_critic_kernel = self.sess.run(self.kernels[1])
            # new_main_critic_bais = self.sess.run(self.biases[0])
            # new_target_critic_bais = self.sess.run(self.biases[1])
            #
            # assert np.alltrue(new_main_critic_kernel == new_target_critic_kernel)
            # assert np.alltrue(new_main_critic_bais == new_main_critic_bais)
            # assert np.count_nonzero(main_critic_kernel - np.diag(np.diagonal(main_critic_kernel))) or len(np.diag(np.diagonal(main_critic_kernel))) == 1
            # new_diagonal = np.diagonal(new_main_critic_kernel)
            # assert np.all([x == new_diagonal[0] for x in new_diagonal])
            #
            # feed_dict={
            #         self.obs1: batch['o_1'],
            #         self.rewards: batch['r']
            # }
            # if self.use_goals:
            #     feed_dict[self.goal1] = batch['g']
            # if not self.use_intrinsic_reward and not self.use_goals:
            #     feed_dict[self.terminals1] = batch['t'].astype('float32')
            #
            # target_Q_new, new_mean_check, new_std_check, pop_is_active_check = self.sess.run(
            #     [self.target_Q, self.ret_rms.mean, self.ret_rms.std, self.ret_rms._pop_is_active],
            #     feed_dict=feed_dict
            # )
            #
            # # print(target_Q_new, target_Q, new_mean, new_std)
            # assert pop_is_active == pop_is_active_check
            # # print("old_mean: {}\nnew mean: {}\nnew_mean_check: {}\n".format(old_mean, new_mean, new_mean_check))
            # assert np.allclose(new_mean, new_mean_check)
            #
            # if not np.allclose(new_std, new_std_check) and MPI.COMM_WORLD.Get_rank() == 1:
            #     print("new_std and new_std_check weren't close:")
            #     print("old_std: {}\nnew std: {}\nnew_std_check: {}\n".format(old_std, new_std, new_std_check))
            # # assert np.allclose(new_std, new_std_check)
            ###

            # TODO: Commented below is for checking if mean/std actually changed in ART adjustment
            # if MPI.COMM_WORLD.Get_rank() == 0:
            #     # print('old std: {} new std: {}'.format(old_std, new_std))
            #     if old_std == new_std:
            #         print('std didnt change')
            #     if old_mean == new_mean:
            #         print('mean didnt change')
            #     pass
            # # assert old_std != new_std
            # # assert old_mean != new_mean
            #
            # if pop_is_active:
            #
            #     # TODO: Commented below is for checking that the POP Matrix head was adjusted properly
            #     if not new_diagonal[0] == diagonal[0] * old_std / new_std:
            #         if MPI.COMM_WORLD.Get_rank() == 0:
            #             print("new diagonal doesnt check out")
            #             print("new diagonal: {}\ndiagonal: {}".format(new_diagonal, diagonal))
            #             print("old std: {}\nnew std: {}".format(old_std, new_std))
            #     # assert new_diagonal[0] == diagonal[0] * old_std / new_std_check
            #     # assert new_main_critic_bais == (old_std * main_critic_bais + old_mean - new_mean) / new_std
            #
            #     # print("pop is active")
            #     if not self.pop_hasnt_been_active:
            #         # TODO: Commented below is for checking if Q Targets changed too much after POPART ajdustment
            #         # print("target_Q: {}\ntarget_Q_new: {}".format(target_Q, target_Q_new))
            #         # assert (np.abs(target_Q - target_Q_new) < 1e-3).all()
            #         if np.all(target_Q != 0):
            #             relative_error = np.mean(np.abs((target_Q - target_Q_new) / target_Q))
            #             if relative_error > 1e-4:
            #                 print("Q targets changed too much after popart adjustment, relative error: {}".format(relative_error))
            #         pass
            #     else:
            #         if MPI.COMM_WORLD.Get_rank() == 0:
            #             print('pop activated')
            #         self.pop_hasnt_been_active = False
            #     # if not (np.abs(target_Q - target_Q_new) < 1e-3).all():
            #     #     print("target Q changed too much")
            #     #     if not (np.abs(target_Q - target_Q_new) < 1e-1).all():
            #     #         print("target Q changed WAAAAAAAAYYYY too much!!!")
            # #
            # #
            # # else:
            # #     print('pop not active')
            # # Also this sanity check takes FOREVER:
            # self.critic_optimizer.check_synced()
            # self.actor_optimizer.check_synced()
            ####

        else:

            feed_dict = {
                self.obs1: batch['o_1'],
                self.rewards: batch['r'],
            }
            if self.use_goals:
                feed_dict[self.goal1] = batch['g']
            elif not self.use_intrinsic_reward:
                feed_dict[self.terminals1] = batch['t'].astype('float32')

            target_Q = self.sess.run(self.target_Q, feed_dict=feed_dict)

        # Get all gradients and perform a synced update.
        if self.mix_external_critic_with_internal is not None:
            ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss, self.combined_actor_grads, self.combined_actor_loss]
        else:
            ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        feed_dict = {
            self.obs0: batch['o'],
            self.actions: batch['u'],
            self.critic_target: target_Q,
        }
        if self.use_goals:
            feed_dict[self.goal0] = batch['g']

        # Todo: Commented below is just for monitoring per batch values
        # predicted_Q = self.sess.run(self.critic_tf, feed_dict=feed_dict)
        # normalized_predicted_Q = self.sess.run(self.normalized_critic_tf, feed_dict=feed_dict)
        # normalized_target_Q = self.sess.run(self.normalized_critic_target_tf, feed_dict=feed_dict)
        # # if MPI.COMM_WORLD.Get_rank() == 0:
        # #     print(predicted_Q)
        # #     print(normalized_predicted_Q)
        # assert not np.all([predicted_Q[0] == Q for Q in predicted_Q])
        # assert not np.all([target_Q[0] == Q for Q in target_Q])
        #
        # # print("target_Q shape: {}".format(target_Q.shape))
        # # exit(0)
        # assert np.all(np.shape(target_Q) == (self.batch_size, 1))
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     logger.record_tabular(self.scope.name + '_train_avg_target_Qs', np.mean(target_Q))
        #     logger.record_tabular(self.scope.name + '_train_avg_predicted_Qs', np.mean(predicted_Q))
        #     logger.record_tabular(self.scope.name + '_train_avg_normalized_target_Qs', np.mean(normalized_target_Q))
        #     logger.record_tabular(self.scope.name + '_train_avg_normalized_predicted_Qs', np.mean(normalized_predicted_Q))
        ####

        if self.mix_external_critic_with_internal is not None:
            actor_grads, actor_loss, critic_grads, critic_loss, combined_actor_grads, combined_actor_loss = self.sess.run(ops, feed_dict=feed_dict)
        else:
            actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict=feed_dict)

        # TODO: Commented below is for sanity check
        # critic_vars_before_update = self.get_critic_var_values()
        # actor_vars_before_update = self.get_actor_var_values()
        ####
        if self.mix_external_critic_with_internal is not None:
            self.combined_actor_optimizer.update(combined_actor_grads, stepsize=self.actor_lr)

        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        # TODO: Commented below is for sanity check
        # critic_vars_after_update = self.get_critic_var_values()
        # actor_vars_after_update = self.get_actor_var_values()
        #
        # assert len(critic_vars_after_update) == len(critic_vars_before_update)
        # assert len(critic_vars_before_update) > 100
        # # assert not np.any(vars_before_update == vars_after_update)
        # assert not np.allclose(critic_vars_before_update, critic_vars_after_update)
        # assert not np.allclose(actor_vars_before_update, actor_vars_after_update)
        # # if np.allclose(actor_vars_before_update, actor_vars_after_update):
        # #     print("\nACTOR VARS DIDNT CHANGE AFTER UPDATE!!!\n")
        ####

        return critic_loss, actor_loss

    def initialize(self):
        agent_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
        init = tf.variables_initializer(var_list=agent_vars)
        self.sess.run(init)
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        if self.mix_external_critic_with_internal is not None:
            self.combined_actor_optimizer.sync()
        if self.use_intrinsic_reward:
            self.dynamics.dynamics_adam.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     print("updated_target_nets")

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        feed_dict={
            self.obs0: self.stats_sample['o'],
            self.actions: self.stats_sample['u'],
        }
        if self.use_goals:
            feed_dict[self.goal0] = self.stats_sample['g']
        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

        #TODO just for debuging print out all actions in batch
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     actions = self.sess.run(self.actor_tf, feed_dict=feed_dict)
        #     print("rank {} Actions:\n".format(MPI.COMM_WORLD.Get_rank()))
        #     print(actions)
        ####

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)

        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        feed_dict = {
            self.obs0: batch['o'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        }
        if self.use_goals:
            feed_dict[self.goal0] = batch['g']
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict=feed_dict)

        mean_distance = self.comm.allreduce(distance, op=MPI.SUM) / self.comm.Get_size()
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

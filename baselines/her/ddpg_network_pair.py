from collections import OrderedDict

import tensorflow as tf
import numpy as np
from baselines.her.util import store_args, nn
from baselines.common.mpi_adam import MpiAdam
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.her.noise import AdaptiveParamNoiseSpec


def _vars(scope):
    res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + '/' + scope)
    assert len(res) > 0
    return res


class DDPGNetworkPair:

    def __init__(self, batch_tf, reuse, input_goals, create_actor_critic_fn, polyak_coeffient, clip_return, clip_pos_returns, gamma, dimu, max_u, o_stats, g_stats, action_l2, ac_hidden, ac_layers, use_layer_norm, use_param_noise, param_noise_stddev):
        if use_param_noise:
            self.param_noise_spec = AdaptiveParamNoiseSpec(initial_stddev=float(param_noise_stddev), desired_action_stddev=float(param_noise_stddev))
        else:
            self.param_noise_spec = None
        # networks
        # with tf.variable_scope('main') as vs:
        #     if reuse:
        #         vs.reuse_variables()
        self.main = create_actor_critic_fn(inputs_tf=batch_tf, input_goals=input_goals, name='main', reuse=reuse, dimu=dimu, max_u=max_u, o_stats=o_stats, g_stats=g_stats, hidden=ac_hidden, layers=ac_layers, use_layer_norm=use_layer_norm, use_param_noise=use_param_noise)
            # vs.reuse_variables()
        target_batch_tf = batch_tf.copy()
        target_batch_tf['o'] = batch_tf['o_2']
        target_batch_tf['g'] = batch_tf['g_2']
        # with tf.variable_scope('target') as vs:
        #     if reuse:
        #         vs.reuse_variables()

        self.target = create_actor_critic_fn(
                inputs_tf=target_batch_tf, input_goals=input_goals, name='target', reuse=reuse, dimu=dimu, max_u=max_u, o_stats=o_stats, g_stats=g_stats, hidden=ac_hidden, layers=ac_layers, use_layer_norm=use_layer_norm, use_param_noise=False)
            # vs.reuse_variables()

        assert len(self.main.all_vars_excluding_param_noise_duplicates) == len(self.target.all_vars_excluding_param_noise_duplicates)

        # loss functions
        self.target_Q_pi_tf = self.target.Q_pi_tf

        # clip_range = (-clip_return, 0. if clip_pos_returns else np.inf)
        # target_tf = tf.clip_by_value(batch_tf['r'] + gamma * target_Q_pi_tf, *clip_range)

        # if input_goals:
        self.target_tf = batch_tf['r'] + gamma * self.target_Q_pi_tf
        # #todo: swap to the above line
        # else:
        #     self.target_tf = batch_tf['r']


        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(self.target_tf) - self.main.Q_tf))
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        # self.pi_loss_tf += action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / max_u))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self.main.critic_vars)
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self.main.actor_vars)
        assert len(_vars('main/Q')) == len(Q_grads_tf)
        assert len(_vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self.main.critic_vars)
        self.pi_grads_vars_tf = zip(pi_grads_tf, self.main.actor_vars)
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self.main.critic_vars)
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self.main.actor_vars)

        # optimizers
        self.Q_adam = MpiAdam(self.main.critic_vars, scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self.main.actor_vars, scale_grad_by_procs=False)

        # polyak averaging

        self.main_vars = self.main.critic_vars + self.main.actor_vars
        self.target_vars = self.target.critic_vars + self.target.actor_vars
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(polyak_coeffient * v[0] + (1. - polyak_coeffient) * v[1]),
                zip(self.target_vars, self.main_vars)))

        self.train_run_vals_tf = OrderedDict([
            ('critic_loss', self.Q_loss_tf),
            ('actor_loss', self.pi_loss_tf),
            ('Q_grad', self.Q_grad_tf),
            ('pi_grad', self.pi_grad_tf)
        ])

        self.optimizers_by_grad_key_tf = OrderedDict([
            ('Q_grad', self.Q_adam),
            ('pi_grad', self.pi_adam),
        ])

    def __getstate__(self):
        #TODO: update exxluded subnames
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        return state
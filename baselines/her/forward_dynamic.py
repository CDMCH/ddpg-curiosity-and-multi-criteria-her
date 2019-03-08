from collections import OrderedDict

import tensorflow as tf
from baselines.her.util import store_args, nn

from baselines.common.mpi_adam import MpiAdam
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)

def _vars(scope):
    res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + '/' + scope)
    assert len(res) > 0
    return res

class ForwardDynamics:
    @store_args
    def __init__(self, inputs_tf, dimo, dimu, max_u, o_stats, hidden, layers):

        self.o_tf = inputs_tf['o']
        self.u_tf = inputs_tf['u']
        self.o2_tf = inputs_tf['o_2']

        # Prepare inputs for actor and critic.
        # o = self.o_stats.normalize(self.o_tf)
        o = self.o_tf

        with tf.variable_scope('forward_dynamics'):
            input = tf.concat(axis=1, values=[o, self.u_tf / self.max_u])
            self.next_state_tf = nn(input, [self.hidden] * self.layers + [dimo])

        # loss functions
        self.per_sample_loss_tf = tf.expand_dims(tf.reduce_mean(tf.square(self.next_state_tf - self.o2_tf), axis=1), axis=1)
        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)
        # self.o1_minus_02_loss_tf = tf.reduce_mean(tf.square(self.o_tf - self.o2_tf))

        grads_tf = tf.gradients(self.mean_loss_tf, _vars('forward_dynamics'))
        assert len(_vars('forward_dynamics')) == len(grads_tf)
        self.grads_vars_tf = zip(grads_tf, _vars('forward_dynamics'))
        self.grad_tf = flatten_grads(grads=grads_tf, var_list=_vars('forward_dynamics'))

        # optimizers
        self.dynamics_adam = MpiAdam(_vars('forward_dynamics'), scale_grad_by_procs=False)

        self.train_run_vals_tf = OrderedDict([
            ('loss_per_sample', self.per_sample_loss_tf),
            ('_loss', self.mean_loss_tf),
            ('_grad', self.grad_tf),
            # ('o1_minus_02_loss', self.o1_minus_02_loss_tf)
        ])

        self.optimizers_by_grad_key_tf = OrderedDict([
            ('_grad', self.dynamics_adam),
        ])

    def __getstate__(self):
        # TODO: update exxluded subnames
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        return state
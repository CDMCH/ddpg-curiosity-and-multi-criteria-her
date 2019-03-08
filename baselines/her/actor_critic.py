import tensorflow as tf
from baselines.her.util import store_args, nn

from baselines import logger


def _get_vars(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def _get_trainable_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def _get_perturbable_vars(scope):
    print("\n\n\n\nVars for scope: {}:\n".format(scope))
    for var in _get_trainable_vars(scope):
        print(var.name)
    return [var for var in _get_trainable_vars(scope) if 'layer_norm' not in var.name]


def _get_perturbed_actor_updates(actor_scope, perturbed_actor_scope, param_noise_stddev_tf, using_layer_norm):
    # set up updates to copy actor variables to param_noise_actor variables
    # add noise to pertinent variables while we do so

    updates = []

    actor_vars = _get_vars(actor_scope)
    perturbed_actor_vars = _get_vars(perturbed_actor_scope)
    perturbable_actor_vars = _get_perturbable_vars(actor_scope)

    assert len(actor_vars) == len(perturbed_actor_vars)
    assert len(perturbable_actor_vars) == len(_get_perturbable_vars(perturbed_actor_scope))
    if using_layer_norm:
        assert len(perturbable_actor_vars) < len(_get_trainable_vars(actor_scope))

    for var, perturbed_var in zip(actor_vars, perturbed_actor_vars):
        if var in perturbable_actor_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(
                tf.assign(perturbed_var,
                          var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev_tf)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor_vars)
    return tf.group(*updates)

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimu, max_u, o_stats, g_stats, hidden, layers, name, reuse, input_goals, use_layer_norm, use_param_noise):
        # print("\n\n\n\n\n\n\n")
        # print(kwargs)
        # print("\n\n\n\n\n\n\n")

        #Todo: remove this line
        # input_goals = True

        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """

        with tf.variable_scope(name, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            self.o_tf = inputs_tf['o']
            self.u_tf = inputs_tf['u']
            self.g_tf = inputs_tf['g']

            # Prepare inputs for actor and critic.
            o = self.o_stats.normalize(self.o_tf)

            if input_goals:
                g = self.g_stats.normalize(self.g_tf)
                input_pi = tf.concat(axis=1, values=[o, g])  # for actor
            else:
                input_pi = o

            # Networks.

            def create_actor():
                return self.max_u * tf.tanh(nn(
                    input_pi, [self.hidden] * self.layers + [self.dimu], use_layer_norm=use_layer_norm))

            with tf.variable_scope('pi'):
                self.actor_scope = tf.get_variable_scope().name
                self.pi_tf = create_actor()

            with tf.variable_scope('Q'):
                self.critic_scope = tf.get_variable_scope().name
                # for policy training
                if input_goals:
                    input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
                else:
                    input_Q = tf.concat(axis=1, values=[o, self.pi_tf / self.max_u])

                self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1], use_layer_norm=False)
                # for critic training
                if input_goals:
                    input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
                else:
                    input_Q = tf.concat(axis=1, values=[o, self.u_tf / self.max_u])
                self._input_Q = input_Q  # exposed for tests
                self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True, use_layer_norm=False)

            if use_param_noise:
                self.param_noise_stddev_tf = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

                with tf.variable_scope('param_noise_pi'):
                    self.param_noise_actor_scope = tf.get_variable_scope().name
                    self.param_noise_pi_tf = create_actor()

                    self.perturb_policy_ops = _get_perturbed_actor_updates(self.actor_scope, self.param_noise_actor_scope,
                                                                           self.param_noise_stddev_tf, using_layer_norm=use_layer_norm)

                with tf.variable_scope('adaptive_param_noise_pi'):
                    self.adaptive_param_noise_actor_scope = tf.get_variable_scope().name
                    adaptive_param_noise_pi_tf = create_actor()

                    self.perturb_adaptive_policy_ops = _get_perturbed_actor_updates(self.actor_scope,
                                                                                    self.adaptive_param_noise_actor_scope,
                                                                                    self.param_noise_stddev_tf, using_layer_norm=use_layer_norm)
                    self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.pi_tf - adaptive_param_noise_pi_tf)))

        print("\nCreated actor critic with scope {}, actor at {}, critic at {}\n".format(self.scope, self.actor_scope, self.critic_scope))



    @property
    def all_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    @property
    def all_vars_excluding_param_noise_duplicates(self):
        return [var for var in self.all_vars if 'param_noise' not in var.name]

    @property
    def actor_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.actor_scope)

    @property
    def critic_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.critic_scope)


import tensorflow as tf
import tensorflow.contrib as tc


def actor(obs, hidden_layer_sizes, nb_actions, use_goals, goal=None, name='actor', layer_norm=True, reuse=False,
          also_return_preactivations=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        if goal is not None:
            x = tf.concat(values=(obs, goal), axis=-1)
        else:
            assert not use_goals
            x = obs

        for i, layer_size in enumerate(hidden_layer_sizes):
            x = tf.layers.dense(x, layer_size)
            if layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

        x = tf.layers.dense(x, nb_actions,
                            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        preactivations = x
        x = tf.nn.tanh(x)

        if also_return_preactivations:
            return x, scope.name, preactivations
        else:
            return x, scope.name


def critic(obs, action, hidden_layer_sizes, use_goals, goal=None, name='critic', layer_norm=True, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        if goal is not None:
            x = tf.concat(values=(obs, action, goal), axis=-1)
        else:
            assert not use_goals
            x = tf.concat(values=(obs, action), axis=-1)

        for i, layer_size in enumerate(hidden_layer_sizes):
            x = tf.layers.dense(x, layer_size)
            if layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

        last_trainable_layer = tf.layers.dense(x, 1)

        pop_art_layer = tf.layers.dense(last_trainable_layer, 1,
                            kernel_initializer=tf.initializers.identity,
                            name='output',
                            trainable=False)

        return pop_art_layer, scope.name, last_trainable_layer

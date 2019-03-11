import ddpg_curiosity_mc_her.common.tf_util as U

import tensorflow as tf
from ddpg_curiosity_mc_her import logger
from ddpg_curiosity_mc_her.common.mpi_adam import MpiAdam


def nn(input, layers_sizes, reuse=None, flatten=False, use_layer_norm=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        norm = tf.contrib.layers.layer_norm if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),

                                reuse=reuse,
                                name=name + '_' + str(i))
        if use_layer_norm and norm:
            print("layer norm")
            input = norm(input, reuse=reuse, scope=name + '_layer_norm_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def _vars(scope):
    res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(res) > 0
    return res


class ForwardDynamics:
    def __init__(self, obs0, action, obs1, clip_norm, hidden, layers, comm):

        logger.info("Using Forward Dynamics")
        assert hidden is not None
        assert layers is not None

        with tf.variable_scope('forward_dynamics'):
            self.dynamics_scope = tf.get_variable_scope().name

            input = tf.concat(values=[obs0, action], axis=-1)
            next_state_tf = nn(input, [hidden] * layers + [obs1.shape[-1]])

        # loss functions
        self.per_sample_loss_tf = tf.expand_dims(tf.reduce_mean(tf.square(next_state_tf - obs1), axis=1), axis=1)
        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)

        self.dynamics_grads = U.flatgrad(self.mean_loss_tf, _vars(self.dynamics_scope), clip_norm=clip_norm)

        # optimizers
        self.dynamics_adam = MpiAdam(_vars(self.dynamics_scope), scale_grad_by_procs=False, comm=comm)


class RandomNetworkDistillation:
    def __init__(self, obs0, action, obs1, clip_norm, hidden, layers):

        logger.info("Using Random Network Distillation")

        rep_size = hidden

        # RND bonus.

        # Random target network.
        # for ph in self.ph_ob.values():
        #     if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
        #         logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        #         xr = ph[:, 1:]
        #         xr = tf.cast(xr, tf.float32)
        #         xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
        #         xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)
        #
        #         xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
        #         xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
        #         xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
        #         rgbr = [to2d(xr)]
        #         X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        with tf.variable_scope('random_network_distillation'):
            self.rnd_scope = tf.get_variable_scope().name
            # Random Target Network

            with tf.variable_scope('target_network'):
                xr = nn(obs1, [hidden] * layers + [rep_size])
                #
                # xr = tf.nn.leaky_relu(fc(obs1, "fc1r", nh=hidden*2, init_scale=np.sqrt(2)))
                # xr = tf.nn.leaky_relu(fc(xr, "fc2r", nh=hidden*2, init_scale=np.sqrt(2)))
                # xr = tf.nn.leaky_relu(fc(xr, "fc3r", nh=hidden, init_scale=np.sqrt(2)))
                # xr = tf.nn.relu(fc(xr, "fc4r", nh=hidden, init_scale=np.sqrt(2)))
                # xr = tf.nn.relu(fc(xr, "fc5r", nh=hidden, init_scale=np.sqrt(2)))
                # xr = fc(xr, "fc6r", nh=rep_size, init_scale=np.sqrt(2))

            with tf.variable_scope('predictor_network'):
                self.predictor_scope = tf.get_variable_scope().name

                # Predictor network.
                # xr_hat = tf.nn.leaky_relu(fc(obs1, "fcr_hat1", nh=hidden*2, init_scale=np.sqrt(2)))
                # xr_hat = tf.nn.leaky_relu(fc(xr_hat, "fcr_hat2", nh=hidden*2, init_scale=np.sqrt(2)))
                # xr_hat = tf.nn.leaky_relu(fc(xr_hat, "fcr_hat3", nh=hidden, init_scale=np.sqrt(2)))
                # xr_hat = tf.nn.relu(fc(xr_hat, "fcr_hat4", nh=hidden, init_scale=np.sqrt(2)))
                # xr_hat = tf.nn.relu(fc(xr_hat, "fcr_hat5", nh=hidden, init_scale=np.sqrt(2)))
                # xr_hat = tf.nn.relu(fc(xr_hat, "fcr_hat6", nh=hidden, init_scale=np.sqrt(2)))
                # xr_hat = tf.nn.relu(fc(xr_hat, "fcr_hat7", nh=hidden, init_scale=np.sqrt(2)))
                # xr_hat = fc(xr_hat, "fcr_hat8", nh=rep_size, init_scale=np.sqrt(2))

                xr_hat = nn(obs1, [hidden] * layers + [rep_size])


        # # Predictor network.
        # for ph in self.ph_ob.values():
        #     if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
        #         logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        #         xrp = ph[:, 1:]
        #         xrp = tf.cast(xrp, tf.float32)
        #         xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
        #         xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)
        #
        #         xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
        #         xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
        #         xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
        #         rgbrp = to2d(xrp)
        #         # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
        #         X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
        #         X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
        #         X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        # self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        # self.max_feat = tf.reduce_max(tf.abs(X_r))
        # self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(xr) - xr_hat), axis=-1, keep_dims=True)

        # targets = tf.stop_gradient(X_r)
        # # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        # self.aux_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)
        # mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        # mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        # self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)
        #

        total_parameters = 0
        for variable in _vars(self.predictor_scope):
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        logger.info("params in target rnd network: {}".format(total_parameters))

        self.feat_var = tf.reduce_mean(tf.nn.moments(xr, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(xr))
        # loss functions
        self.per_sample_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(xr) - xr_hat), axis=-1, keepdims=True)
        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)

        self.dynamics_grads = U.flatgrad(self.mean_loss_tf, _vars(self.predictor_scope), clip_norm=clip_norm)

        # optimizers
        self.dynamics_adam = MpiAdam(_vars(self.predictor_scope), scale_grad_by_procs=False)

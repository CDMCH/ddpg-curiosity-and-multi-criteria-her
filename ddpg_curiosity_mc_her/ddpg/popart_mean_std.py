from mpi4py import MPI
import tensorflow as tf, ddpg_curiosity_mc_her.common.tf_util as U, numpy as np
from functools import reduce
import math

class PopArtMeanStd(object):

    def __init__(self, sess, kernels, biases, comm, beta=1e-5, epsilon=1e-2, stable_rate=0.005, min_steps=100000):

        self.mean = tf.get_variable(
            dtype=tf.float32,
            shape=(),
            initializer=tf.constant_initializer(0.0),
            name="running_mean", trainable=False)
        self._mean_of_sq = tf.get_variable(
            dtype=tf.float32,
            shape=(),
            initializer=tf.constant_initializer(0.0),
            name="running_mean_of_sq", trainable=False)
        self._step = tf.get_variable(
            dtype=tf.int64,
            shape=(),
            initializer=tf.constant_initializer(1),
            name="step", trainable=False)
        self._pop_is_active = tf.get_variable(
            dtype=tf.int8,
            shape=(),
            initializer=tf.constant_initializer(0),
            name="pop_is_active", trainable=False)

        self.new_mean_placeholder = tf.placeholder(dtype=self.mean.dtype, shape=self.mean.shape)
        self.new_mean_of_sq_placeholder = tf.placeholder(dtype=self._mean_of_sq.dtype, shape=self._mean_of_sq.shape)
        self.new_step_placeholder = tf.placeholder(dtype=self._step.dtype, shape=self._step.shape)
        self.new_pop_is_active_placeholder = tf.placeholder(dtype=self._pop_is_active.dtype, shape=self._pop_is_active.shape)

        self.beta = np.float32(beta)
        self.epsilon = np.float32(epsilon)
        self.stable_rate = np.float32(stable_rate)
        self.min_steps = np.int32(min_steps)
        self.comm = comm

        self.std = tf.sqrt(tf.maximum(self._mean_of_sq - tf.square(self.mean), self.epsilon))


        # self.update_kernals_biases = U.function([new_kernel_vals, new_bias_vals, newcount], [],
        #     updates=[tf.assign_add(self._sum, newsum),
        #              tf.assign_add(self._sumsq, newsumsq),
        #              tf.assign_add(self._count, newcount)])

        self.sess = sess

        assert isinstance(kernels, list)
        assert isinstance(biases, list)
        assert len(kernels) == len(biases)
        assert len(kernels) == 2
        self.kernels = kernels
        self.biases = biases

        self.new_kernel_placeholders = []
        for kernel in self.kernels:
            self.new_kernel_placeholders.append(tf.placeholder(dtype=kernel.dtype, shape=kernel.shape))
        self.new_bias_placeholders = []
        for bias in self.biases:
            self.new_bias_placeholders.append(tf.placeholder(dtype=bias.dtype, shape=bias.shape))


        self.assign_new_kernels_biases_ops = [var.assign(placeholder) for var, placeholder in zip(self.kernels, self.new_kernel_placeholders)]
        self.assign_new_kernels_biases_ops += [var.assign(placeholder) for var, placeholder in zip(self.biases, self.new_bias_placeholders)]

        assert len(self.assign_new_kernels_biases_ops) == len(kernels) * 2
        assert len(self.assign_new_kernels_biases_ops) > 0

        self.assign_new_rms_ops = [
            self.mean.assign(self.new_mean_placeholder),
            self._mean_of_sq.assign(self.new_mean_of_sq_placeholder),
            self._step.assign(self.new_step_placeholder),
            self._pop_is_active.assign(self.new_pop_is_active_placeholder)
        ]

    # def update(self, x):
    #     x = x.astype('float64')
    #     n = int(np.prod(self.shape))
    #     totalvec = np.zeros(n*2+1, 'float64')
    #     addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])
    #     MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
    #     self.incfiltparams(totalvec[0:n].reshape(self.shape), totalvec[n:2*n].reshape(self.shape), totalvec[2*n])

    # def de_normalize_np(self, x):
    #     """
    #     Converts previously normalized data into original values.
    #     """
    #     std_dev = np.sqrt(self._mean_of_sq - np.square(self.mean))
    #     return (x * (std_dev if std_dev > 0 else self.epsilon)
    #             + self.mean)
    #
    # def normalize_np(self, x):
    #     std_dev = np.sqrt(self._mean_of_sq - np.square(self.mean))
    #     result = ((x - self.mean) / (std_dev if std_dev > 0 else self.epsilon))
    #     return result

    def pop_art_update(self, x: np.ndarray):
        """
        Performs ART (Adaptively Rescaling Targets) update,
        adjusting normalization parameters with respect to new targets x.
        Updates running mean, mean of squares and returns
        new mean and standard deviation for later use.
        """

        #TODO: sanity check assertion
        # assert len(x.shape) == 1, "Must be 1D (batch_size,)"
        ###

        beta = self.beta

        (old_online_mean, old_online_mean_of_square, step, pop_is_active) = self.sess.run(
            [self.mean, self._mean_of_sq, self._step, self._pop_is_active]
        )

        # Average local batch values, then average local values to get global batch averages
        local_x_means = np.stack([x.mean(axis=0), np.square(x).mean(axis=0)], axis=0)

        #TODO: sanity check assertion
        # assert len(local_x_means.shape) == 1
        # assert local_x_means.shape[0] == 2
        ###

        x_means = np.zeros_like(local_x_means)

        self.comm.Allreduce(sendbuf=local_x_means, recvbuf=x_means, op=MPI.SUM)
        x_means /= self.comm.Get_size()

        #TODO: sanity check assertion
        # assert len(x_means.shape) == 1
        # assert x_means.shape[0] == 2
        ###

        # Get global batch sample count
        batch_sample_count_local = np.asarray(x.shape[0])
        batch_sample_count = np.zeros_like(batch_sample_count_local)
        self.comm.Allreduce(sendbuf=batch_sample_count_local, recvbuf=batch_sample_count, op=MPI.SUM)
        batch_sample_count = np.squeeze(batch_sample_count)



        #TODO: sanity check assertion
        # batch_sample_count_old_for_sanity_check =np.squeeze(batch_sample_count_local)
        # assert batch_sample_count_old_for_sanity_check * MPI.COMM_WORLD.Get_size() == batch_sample_count
        ###

        def update_avg(old_avg, new_sample, sample_size):
            """
            Update rule for running estimations,
            dynamically adjusting sensitivity with every time step
            to new data (see Eq. 10 in the paper).
            """

            adj_beta = np.float32(min(((beta / (1 - (1 - beta)**step)) * sample_size), 1))

            # print("Adjusted_beta: {}, beta: {}, step: {}, sample_size: {}".format(adj_beta, beta, step, sample_size))

            #TODO: sanity check assertion
            # assert isinstance(beta, np.float32)
            # assert sample_size >= 1
            # assert step >= 1
            #
            # assert not (math.isnan(adj_beta))
            # assert not (math.isnan(old_avg))
            # assert not (math.isnan(new_sample))
            # assert not (math.isnan(sample_size))
            #
            # assert not np.isclose(adj_beta, 0)
            # assert adj_beta != 0.0
            # assert adj_beta > 0.0
            # assert adj_beta <= 1
            #
            # assert isinstance(adj_beta, np.float32)
            #
            # # print("types\nadj_beta:{}\nold_avg:{}\nnew_sample:{}\n".format(type(adj_beta), type(old_avg), type(new_sample)))
            #
            # # print("adj_beta: {}".format(adj_beta))
            # assert not isinstance(old_avg, np.float64)
            # assert not isinstance(new_sample, np.float64)
            ###


            new_avg = (np.float32(1) - adj_beta) * old_avg + adj_beta * new_sample

            #TODO: sanity check assertion
            # assert not isinstance((np.float32(1) - adj_beta), np.float64)
            # assert isinstance((np.float32(1) - adj_beta), np.float32)
            # assert isinstance(old_avg, np.float32)
            # assert not isinstance((np.float32(1) - adj_beta) * old_avg, np.float64)
            # assert not isinstance(adj_beta * new_sample, np.float64)
            # assert not isinstance(new_avg, np.float64)
            ###

            return new_avg

        online_mean = update_avg(old_avg=old_online_mean, new_sample=x_means[0], sample_size=batch_sample_count)
        online_mean_of_square = update_avg(old_avg=old_online_mean_of_square, new_sample=x_means[1], sample_size=batch_sample_count)

        # TODO: sanity check assertion
        # assert not (math.isnan(online_mean_of_square))
        # assert not (math.isnan(online_mean))
        ###

        step += batch_sample_count

        old_std_dev = np.sqrt(old_online_mean_of_square - np.square(old_online_mean))
        #
        # print("x_means: {}".format(x_means))
        # print("mean of square: {}".format(online_mean_of_square))
        # print("mean: {}".format(online_mean))
        # print("mean_squared: {}".format(np.square(online_mean)))
        #
        # exit()
        std_dev = np.sqrt(max(online_mean_of_square - np.square(online_mean), self.epsilon))
        old_std_dev = old_std_dev if old_std_dev > 0 else std_dev

        # TODO: sanity check assertion
        # assert not (math.isnan(std_dev))
        # assert not np.isclose(std_dev, 0)
        ###


        # Performing POP (Preserve the Output Precisely) update
        # but only if we are not in the beginning of the training
        # when both mean and std_dev are close to zero or still
        # stabilizing. Otherwise POP kernel (W) and bias (b) can
        # become very large and cause numerical instability.
        std_is_stable = (step > self.min_steps and np.abs(1 - old_std_dev / std_dev) < self.stable_rate)

        if int(pop_is_active) == 1 or (std_dev > self.epsilon and std_is_stable):

            kernel_vals = self.sess.run(self.kernels)
            bias_vals = self.sess.run(self.biases)

            #TODO: sanity check assertion
            # debug_old_kernel_vals = kernel_vals.copy()
            #
            # #these check dont necessariy need to hold in all cases
            # assert np.all(kernel_vals[0] == kernel_vals[1])
            # assert np.all(bias_vals[0] == bias_vals[1])
            ###


            for i in range(len(kernel_vals)):
                old_kernel = kernel_vals[i]
                old_bias = bias_vals[i]

                kernel_vals[i] = old_std_dev * old_kernel / std_dev
                bias_vals[i] = (old_std_dev * old_bias + old_online_mean - online_mean) / std_dev

                # TODO: sanity check assertion
                # if not np.isclose(old_std_dev, std_dev):
                #     # assert not np.all(kernel_vals[i] == old_kernel)
                #     if np.all(kernel_vals[i] == old_kernel):
                #         print("pop kernal didnt change even though art stddev did change")
                ###

                # print("change in kernel: x{} change in bias: difference of {}".format(old_std_dev/std_dev, bias_vals[i] - old_bias))

            #TODO: sanity check assertion
            # assert np.all(kernel_vals[0] == kernel_vals[1])
            # assert np.all(bias_vals[0] == bias_vals[1])
            ###

            pop_is_active = 1

            feed_dict = {kernel_placeholder: kernel_val for kernel_placeholder, kernel_val in zip(self.new_kernel_placeholders, kernel_vals)}
            feed_dict.update({bias_placeholder: bias_val for bias_placeholder, bias_val in zip(self.new_bias_placeholders, bias_vals)})
            self.sess.run(self.assign_new_kernels_biases_ops, feed_dict=feed_dict)

            #TODO: sanity check assertion
            # debug_new_kernel_vals = self.sess.run(self.kernels)
            # if not np.isclose(old_std_dev, std_dev):
            #     # print("old_std: {}, new_std:{}".format(old_std_dev, std_dev))
            #
            #     for i in range(len(debug_old_kernel_vals)):
            #         assert not np.all(debug_old_kernel_vals[i] == debug_new_kernel_vals[i])
            #
            # # print("old_std: {}, new_std:{}".format(old_std_dev, std_dev))
            # # exit()
            # for i in range(len(debug_old_kernel_vals)):
            #     assert np.allclose(debug_new_kernel_vals[i], debug_old_kernel_vals[i] * old_std_dev / std_dev)
            ###

        # Saving updated parameters into graph variables
        # var_updates = [
        #     self.mean.assign(online_mean),
        #     self._mean_of_sq.assign(online_mean_of_square),
        #     self._step.assign(step),
        #     self._pop_is_active.assign(pop_is_active)
        # ]

        feed_dict = {
            self.new_mean_placeholder: online_mean,
            self.new_mean_of_sq_placeholder: online_mean_of_square,
            self.new_step_placeholder: step,
            self.new_pop_is_active_placeholder: pop_is_active
        }

        self.sess.run(self.assign_new_rms_ops, feed_dict=feed_dict)

        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     for name, value in locals().copy().items():
        #         print(name, type(value), value)
        # exit()

        return pop_is_active, online_mean, std_dev

    # def update_and_normalize(self, x: np.ndarray):
    #     """
    #     Normalizes given tensor `x` and updates parameters associated
    #     with PopArt: running means (art) and network's output scaling (pop).
    #     """
    #     mean, std_dev = self.pop_art_update(x)
    #     normalized_input = ((x - mean) / (std_dev if std_dev > 0 else self.epsilon))
    #     return normalized_input, mean, std_dev

# @U.in_session
# def test_runningmeanstd():
#     for (x1, x2, x3) in [
#         (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
#         (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
#         ]:
#
#         rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])
#         U.initialize()
#
#         x = np.concatenate([x1, x2, x3], axis=0)
#         ms1 = [x.mean(axis=0), x.std(axis=0)]
#         rms.update(x1)
#         rms.update(x2)
#         rms.update(x3)
#         ms2 = [rms.mean.eval(), rms.std.eval()]
#
#         assert np.allclose(ms1, ms2)
#
# @U.in_session
# def test_dist():
#     np.random.seed(0)
#     p1,p2,p3=(np.random.randn(3,1), np.random.randn(4,1), np.random.randn(5,1))
#     q1,q2,q3=(np.random.randn(6,1), np.random.randn(7,1), np.random.randn(8,1))
#
#     # p1,p2,p3=(np.random.randn(3), np.random.randn(4), np.random.randn(5))
#     # q1,q2,q3=(np.random.randn(6), np.random.randn(7), np.random.randn(8))
#
#     comm = MPI.COMM_WORLD
#     assert comm.Get_size()==2
#     if comm.Get_rank()==0:
#         x1,x2,x3 = p1,p2,p3
#     elif comm.Get_rank()==1:
#         x1,x2,x3 = q1,q2,q3
#     else:
#         assert False
#
#     rms = RunningMeanStd(epsilon=0.0, shape=(1,))
#     U.initialize()
#
#     rms.update(x1)
#     rms.update(x2)
#     rms.update(x3)
#
#     bigvec = np.concatenate([p1,p2,p3,q1,q2,q3])
#
#     def checkallclose(x,y):
#         print(x,y)
#         return np.allclose(x,y)
#
#     assert checkallclose(
#         bigvec.mean(axis=0),
#         rms.mean.eval(),
#     )
#     assert checkallclose(
#         bigvec.std(axis=0),
#         rms.std.eval(),
#     )
#
#
# if __name__ == "__main__":
#     # Run with mpirun -np 2 python <filename>
#     test_dist()

import os.path
import subprocess
import tempfile
import warnings
from functools import reduce
from typing import Tuple, NamedTuple, List, Union

import h5py
import numpy as np
from PIL import Image
from keras.models import Model, Sequential
from keras.layers import Layer
from keras import optimizers
# noinspection PyPep8Naming
from keras import backend as K


class PopArtLayer(Layer):
    """
    Automatic network output scale adjuster, which is supposed to keep
    the output of the network up to date as we keep updating moving
    average and variance of discounted returns.

    Part of the PopArt algorithm described in DeepMind's paper
    "Learning values across many orders of magnitude"
    (https://arxiv.org/abs/1602.07714)
    """
    def __init__(self, beta=1e-4, epsilon=1e-4, stable_rate=0.1,
                 min_steps=1000, **kwargs):
        """
        :param beta: a value in range (0..1) controlling sensitivity to changes
        :param epsilon: a minimal possible value replacing standard deviation
                if the original one is zero.
        :param stable_rate: Pop-part of the algorithm will kick in only when
            the amplitude of changes in standard deviation will drop
            to this value (stabilizes). This protects pop-adjustments from
            being activated too soon, which would lead to weird values
            of `W` and `b` and numerical instability.
        :param min_steps: Minimal number of steps before it even begins
            possible for Pop-part to become activated (an extra precaution
            in addition to the `stable_rate`).
        :param kwargs: any extra Keras layer parameters, like name, etc.
        """
        self.beta = beta
        self.epsilon = epsilon
        self.stable_rate = stable_rate
        self.min_steps = min_steps
        super().__init__(**kwargs)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel', shape=(), dtype='float32',
            initializer='ones', trainable=False)
        self.bias = self.add_weight(
            name='bias', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.mean = self.add_weight(
            name='mean', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.mean_of_square = self.add_weight(
            name='mean_of_square', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.step = self.add_weight(
            name='step', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        self.pop_is_active = self.add_weight(
            name='pop_is_active', shape=(), dtype='float32',
            initializer='zeros', trainable=False)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.kernel * inputs + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

    def de_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Converts previously normalized data into original values.
        """
        online_mean, online_mean_of_square = K.batch_get_value(
            [self.mean, self.mean_of_square])
        std_dev = np.sqrt(online_mean_of_square - np.square(online_mean))
        return (x * (std_dev if std_dev > 0 else self.epsilon)
                + online_mean)

    def pop_art_update(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Performs ART (Adaptively Rescaling Targets) update,
        adjusting normalization parameters with respect to new targets x.
        Updates running mean, mean of squares and returns
        new mean and standard deviation for later use.
        """
        assert len(x.shape) == 2, "Must be 2D (batch_size, time_steps)"
        beta = self.beta
        (old_kernel, old_bias, old_online_mean,
         old_online_mean_of_square, step, pop_is_active) = K.batch_get_value(
            [self.kernel, self.bias, self.mean,
             self.mean_of_square, self.step, self.pop_is_active])

        def update_rule(old, new):
            """
            Update rule for running estimations,
            dynamically adjusting sensitivity with every time step
            to new data (see Eq. 10 in the paper).
            """
            nonlocal step
            step += 1
            adj_beta = beta / (1 - (1 - beta)**step)
            return (1 - adj_beta) * old + adj_beta * new

        x_means = np.stack([x.mean(axis=0), np.square(x).mean(axis=0)], axis=1)
        # Updating normalization parameters (for ART)
        online_mean, online_mean_of_square = reduce(
            update_rule, x_means,
            np.array([old_online_mean, old_online_mean_of_square]))
        old_std_dev = np.sqrt(
            old_online_mean_of_square - np.square(old_online_mean))
        std_dev = np.sqrt(online_mean_of_square - np.square(online_mean))
        old_std_dev = old_std_dev if old_std_dev > 0 else std_dev
        # Performing POP (Preserve the Output Precisely) update
        # but only if we are not in the beginning of the training
        # when both mean and std_dev are close to zero or still
        # stabilizing. Otherwise POP kernel (W) and bias (b) can
        # become very large and cause numerical instability.
        std_is_stable = (
            step > self.min_steps
            and np.abs(1 - old_std_dev / std_dev) < self.stable_rate)
        if (int(pop_is_active) == 1 or
                (std_dev > self.epsilon and std_is_stable)):
            new_kernel = old_std_dev * old_kernel / std_dev
            new_bias = (
                (old_std_dev * old_bias + old_online_mean - online_mean)
                / std_dev)
            pop_is_active = 1
        else:
            new_kernel, new_bias = old_kernel, old_bias
        # Saving updated parameters into graph variables
        var_update = [
            (self.kernel, new_kernel),
            (self.bias, new_bias),
            (self.mean, online_mean),
            (self.mean_of_square, online_mean_of_square),
            (self.step, step),
            (self.pop_is_active, pop_is_active)]
        K.batch_set_value(var_update)
        return online_mean, std_dev

    def update_and_normalize(self, x: np.ndarray) -> Tuple[np.ndarray,
                                                           float, float]:
        """
        Normalizes given tensor `x` and updates parameters associated
        with PopArt: running means (art) and network's output scaling (pop).
        """
        mean, std_dev = self.pop_art_update(x)
        result = ((x - mean) / (std_dev if std_dev > 0 else self.epsilon))
        return result, mean, std_dev
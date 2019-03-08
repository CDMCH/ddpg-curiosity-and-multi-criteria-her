import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape, intrinsic_reward_vals_only):
        self.limit = limit
        self.intrinsic_reward_vals_only = intrinsic_reward_vals_only

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.observations1 = RingBuffer(limit, shape=observation_shape)

        if not self.intrinsic_reward_vals_only:
            self.rewards = RingBuffer(limit, shape=(1,))
            self.terminals1 = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)

        result = {
            'o': array_min2d(obs0_batch),
            'o_1': array_min2d(obs1_batch),
            'u': array_min2d(action_batch),
        }

        if not self.intrinsic_reward_vals_only:
            reward_batch = self.rewards.get_batch(batch_idxs)
            terminal1_batch = self.terminals1.get_batch(batch_idxs)

            result.update({
                'r': array_min2d(reward_batch),
                't': array_min2d(terminal1_batch)
            })

        return result

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        # batch_shapes = {}
        # for key, value in episode_batch.items():
        #     batch_shapes[key] = value.shape
        # print("\n\nbatch shapes:\n{}".format(batch_shapes))
        # exit(0)

        obs0_episode = episode_batch['o'][:, :-1, :]
        action_episode = episode_batch['u']
        obs1_episode = episode_batch['o'][:, 1:, :]

        if not self.intrinsic_reward_vals_only:
            reward_episode = episode_batch['r']
            terminal1_episode = episode_batch['t']

        episode_length = action_episode.shape[1]

        for episode in range(batch_size):
            for transition in range(episode_length):
                self.append(
                    obs0=obs0_episode[episode, transition, ...],
                    action=action_episode[episode, transition, ...],
                    obs1=obs1_episode[episode, transition, ...],
                    reward=reward_episode[episode, transition, ...] if not self.intrinsic_reward_vals_only else None,
                    terminal1=terminal1_episode[episode, transition, ...] if not self.intrinsic_reward_vals_only else None
                )

    def append(self, obs0, action, obs1, reward, terminal1, training=True):
        if not training:
            return
        
        self.observations0.append(obs0)
        self.actions.append(action)
        self.observations1.append(obs1)
        if not self.intrinsic_reward_vals_only:
            self.rewards.append(reward)
            self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)

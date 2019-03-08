import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun=None, sub_goal_divisions=None):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals, None to keep reward as is.
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    # def _sample_her_transitions(episode_batch, batch_size_in_transitions):
    #     """episode_batch is {key: array(buffer_size x T x dim_key)}
    #     """
    #     T = episode_batch['u'].shape[1]
    #     rollout_batch_size = episode_batch['u'].shape[0]
    #     batch_size = batch_size_in_transitions
    #
    #     # Select which episodes and time steps to use.
    #     episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    #     t_samples = np.random.randint(T, size=batch_size)
    #     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
    #                    for key in episode_batch.keys()}
    #
    #     if future_p > 0:
    #         # Select future time indexes proportional with probability future_p. These
    #         # will be used for HER replay by substituting in future goals.
    #         her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
    #         future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    #         future_offset = future_offset.astype(int)
    #         future_t = (t_samples + 1 + future_offset)[her_indexes]
    #
    #         # Replace goal with achieved goal but only for the previously-selected
    #         # HER transitions (as defined by her_indexes). For the other transitions,
    #         # keep the original goal.
    #         future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
    #         transitions['g'][her_indexes] = future_ag
    #
    #     if reward_fun is not None:
    #         # Reconstruct info dictionary for reward  computation.
    #         info = {}
    #         for key, value in transitions.items():
    #             if key.startswith('info_'):
    #                 info[key.replace('info_', '')] = value
    #
    #         # Re-compute reward since we may have substituted the goal.
    #         reward_params = {k: transitions[k] for k in ['ag_1', 'g']}
    #         reward_params['info'] = info
    #         transitions['r'] = reward_fun(**reward_params)
    #
    #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
    #                    for k in transitions.keys()}

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        if future_p > 0:

            # Allow HER operation to be done on subsections of a goal independently. If sub_goal_divisions is not
            # specified, perform vanilla HER operation on the entire goal
            if sub_goal_divisions is None:
                sub_goal_divisions_to_use = [range(transitions['g'].shape[1])]
            else:
                sub_goal_divisions_to_use = sub_goal_divisions
                assert sum([len(elem) for elem in sub_goal_divisions_to_use]) + 3 == episode_batch['g'].shape[2]

            # # TODO
            # # TODO  -- REMOVE USING JUST ONE SUB GOAL!!!!!
            # # TODO
            # # TODO
            #
            # sub_goal_divisions_to_use = [sub_goal_divisions_to_use[-1]]
            # assert len(sub_goal_divisions_to_use) == 1
            #
            # # TODO
            # # TODO

            for sub_goal_division in sub_goal_divisions_to_use:
                # Select future time indexes proportional with probability future_p. These
                # will be used for HER replay by substituting in future goals.

                # Choose which transitions from the batch we will alter
                her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

                # For each transition, choose a time offset between the max episode length and the time of the sample
                future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                future_offset = future_offset.astype(int)

                # Get the future times to sample goals from (transitions not chosen for altering have original times)
                future_t = (t_samples + 1 + future_offset)[her_indexes]

                # Replace goal with achieved goal but only for the previously-selected
                # HER transitions (as defined by her_indexes). For the other transitions,
                # keep the original goal.

                future_achieved_goals = episode_batch['ag'][episode_idxs[her_indexes], future_t]

                old_transition = transitions['g'].copy()

                transition_goals = transitions['g'][her_indexes]
                transition_goals[:, sub_goal_division] = future_achieved_goals[:, sub_goal_division]
                transitions['g'][her_indexes] = transition_goals

                if batch_size == 5:
                    print("T: {}".format(T))
                    print("batch_size: {}".format(batch_size_in_transitions))
                    print("transitions['g'] shape: {}".format(transitions['g'].shape))
                    print("transitions['g']:\n{}".format(old_transition))
                    print("her indexes: {}".format(her_indexes))
                    print("future_achieved_goals_shape: {}".format(future_achieved_goals.shape))
                    print("future_achieved_goals[:, sub_goal_division] shape: {}".format(future_achieved_goals[:, sub_goal_division].shape))
                    print("future_achieved_goals: {}".format(future_achieved_goals))
                    print("future_achieved_goals[:, sub_goal_division]: {}".format(future_achieved_goals[:, sub_goal_division]))
                    print("transition_goals: {}".format(transition_goals))
                    print("modified transitions['g']: {}".format(transitions['g']))
                    print("\n\n")
        if batch_size == 5:
            exit(0)

        if reward_fun is not None:
            # Reconstruct info dictionary for reward computation.
            info = {}
            for key, value in transitions.items():
                if key.startswith('info_'):
                    info[key.replace('info_', '')] = value

            # Re-compute reward since we may have substituted the goal.
            reward_params = {k: transitions[k] for k in ['ag_1', 'g']}
            reward_params['info'] = info
            transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions




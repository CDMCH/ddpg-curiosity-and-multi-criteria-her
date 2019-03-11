import numpy as np
from mpi4py import MPI


def greedy_exploit(agents):
    assert 'exploit' in agents.keys()

    def get_actions(observation, goal, compute_Q):
        return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=False, compute_Q=compute_Q)

    return get_actions


def noisy_exploit(agents):
    assert 'exploit' in agents.keys()

    def get_actions(observation, goal, compute_Q):
        return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    return get_actions


def epsilon_greedy_noisy_exploit(agents):
    # Used in reference HER implementation along with a normal action noise of std 0.2
    assert 'exploit' in agents.keys()
    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.3

    def get_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    return get_actions


def greedy_explore(agents):
    assert 'explore' in agents.keys()

    def get_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=False, compute_Q=compute_Q)

    return get_actions


def noisy_explore(agents):
    assert 'explore' in agents.keys()

    def get_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    return get_actions


def epsilon_greedy_noisy_explore(agents):
    assert 'explore' in agents.keys()
    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.3

    def get_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    return get_actions


def epsilon_0p1_greedy_noisy_explore(agents):
    assert 'explore' in agents.keys()
    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.1

    def get_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    return get_actions


def random(agents):
    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range

    def get_actions(observation, goal, compute_Q):
        return np.random.uniform(low=action_range[0], high=action_range[1], size=(observation.shape[0], *action_shape))

    return get_actions


def noisy_exploit_with_one_noisy_explore_worker(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    # Must have at least two workers for this policy selection to make sense
    assert MPI.COMM_WORLD.Get_size() > 1

    def get_exploit_actions(observation, goal, compute_Q):
        return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    def get_explore_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        return get_explore_actions

    return get_exploit_actions


def epsilon_greedy_noisy_exploit_with_one_noisy_explore_worker(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    # Must have at least two workers for this policy selection to make sense
    assert MPI.COMM_WORLD.Get_size() > 1

    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.3

    def get_exploit_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    def get_explore_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        return get_explore_actions

    return get_exploit_actions


def epsilon_greedy_noisy_exploit_with_two_noisy_explore_workers(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    # Must have at least three workers for this policy selection to make sense
    assert MPI.COMM_WORLD.Get_size() > 2

    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.3

    def get_exploit_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    def get_explore_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0 or rank == 1:
        return get_explore_actions

    return get_exploit_actions


def epsilon_greedy_noisy_exploit_with_four_noisy_explore_workers(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    # Must have at least three workers for this policy selection to make sense
    assert MPI.COMM_WORLD.Get_size() > 2

    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.3

    def get_exploit_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    def get_explore_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank in [0, 1, 2, 3]:
        return get_explore_actions

    return get_exploit_actions


def half_epsilon_greedy_noisy_exploit_half_epsilon_greedy_noisy_explore(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    # Must have at least two workers for this policy selection to make sense
    assert MPI.COMM_WORLD.Get_size() >= 2

    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.3

    def get_exploit_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    def get_explore_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    rank = MPI.COMM_WORLD.Get_rank()

    if rank % 2 == 0:
        return get_explore_actions

    return get_exploit_actions


def half_noisy_exploit_half_noisy_explore(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    # Must have at least two workers for this policy selection to make sense
    assert MPI.COMM_WORLD.Get_size() >= 2

    agent = list(agents.values())[0]

    def get_exploit_actions(observation, goal, compute_Q):
        return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    def get_explore_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank % 2 == 0:
        return get_explore_actions

    return get_exploit_actions


def epsilon_greedy_noisy_exploit_with_one_greedy_explore_worker(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    # Must have at least two workers for this policy selection to make sense
    assert MPI.COMM_WORLD.Get_size() > 1

    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.3

    def get_exploit_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    def get_explore_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=False, compute_Q=compute_Q)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        return get_explore_actions

    return get_exploit_actions

def epsilon_greedy_epsilon_explore_noisy_exploit(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    random_epsilon = 0.1
    explore_epsilon = 0.2

    assert random_epsilon + explore_epsilon <= 1

    def get_actions(observation, goal, compute_Q):
        choice = np.random.rand()
        if choice > random_epsilon + explore_epsilon:
            return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        elif choice > random_epsilon:
            return agents['explore'].pi(obs=observation, goal=goal, apply_noise=False, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    return get_actions


def epsilon_greedy_noisy_explore_with_four_noisy_exploit_workers(agents):
    assert 'exploit' in agents.keys()
    assert 'explore' in agents.keys()

    # Must have at least three workers for this policy selection to make sense
    assert MPI.COMM_WORLD.Get_size() > 4

    agent = list(agents.values())[0]
    action_shape = agent.action_shape
    action_range = agent.action_range
    epsilon = 0.3

    def get_exploit_actions(observation, goal, compute_Q):
        if np.random.rand() > epsilon:
            return agents['exploit'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)
        else:
            return np.random.uniform(low=action_range[0], high=action_range[1],
                                     size=(observation.shape[0], *action_shape))

    def get_explore_actions(observation, goal, compute_Q):
        return agents['explore'].pi(obs=observation, goal=goal, apply_noise=True, compute_Q=compute_Q)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank in [0, 1, 2, 3]:
        return get_exploit_actions

    return get_explore_actions


def get_policy_fn(name, agents):

    if name == 'greedy_exploit':
        return greedy_exploit(agents=agents)
    elif name == 'noisy_exploit':
        return noisy_exploit(agents=agents)
    elif name == 'epsilon_greedy_noisy_exploit':
        return epsilon_greedy_noisy_exploit(agents=agents)
    elif name == 'greedy_explore':
        return greedy_explore(agents=agents)
    elif name == 'noisy_explore':
        return noisy_explore(agents=agents)
    elif name == 'epsilon_greedy_noisy_explore':
        return epsilon_greedy_noisy_explore(agents=agents)
    elif name == 'random':
        return random(agents=agents)
    elif name == 'noisy_exploit_with_one_noisy_explore_worker':
        return noisy_exploit_with_one_noisy_explore_worker(agents=agents)
    elif name == 'epsilon_greedy_noisy_exploit_with_one_greedy_explore_worker':
        return epsilon_greedy_noisy_exploit_with_one_greedy_explore_worker(agents=agents)
    elif name == 'epsilon_greedy_noisy_exploit_with_one_noisy_explore_worker':
        return epsilon_greedy_noisy_exploit_with_one_noisy_explore_worker(agents=agents)
    elif name == 'epsilon_greedy_noisy_exploit_with_two_noisy_explore_workers':
        return epsilon_greedy_noisy_exploit_with_two_noisy_explore_workers(agents=agents)
    elif name == 'epsilon_greedy_noisy_exploit_with_four_noisy_explore_workers':
        return epsilon_greedy_noisy_exploit_with_four_noisy_explore_workers(agents=agents)
    elif name == 'epsilon_greedy_epsilon_explore_noisy_exploit':
        return epsilon_greedy_epsilon_explore_noisy_exploit(agents=agents)
    elif name == 'epsilon_greedy_noisy_explore_with_four_noisy_exploit_workers':
        return epsilon_greedy_noisy_explore_with_four_noisy_exploit_workers(agents=agents)
    elif name == 'half_epsilon_greedy_noisy_exploit_half_epsilon_greedy_noisy_explore':
        return half_epsilon_greedy_noisy_exploit_half_epsilon_greedy_noisy_explore(agents=agents)
    elif name == 'epsilon_0p1_greedy_noisy_explore':
        return epsilon_0p1_greedy_noisy_explore(agents=agents)
    elif name == 'half_noisy_exploit_half_noisy_explore':
        return half_noisy_exploit_half_noisy_explore(agents=agents)
    else:
        raise ValueError('Unknown policy function name: {}'.format(name))

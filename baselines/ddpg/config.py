import gym
import os
import ast
from collections import OrderedDict
from baselines import logger
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.her import make_sample_her_transitions
from baselines.ddpg.noise import *
from baselines.ddpg.replay_buffer import ReplayBuffer
from baselines.ddpg.rollout import RolloutWorker
from baselines.ddpg.dynamics_loss_mapping import DynamicsLossMapper
from mpi4py import MPI


DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
    'boxpush-v0': {
        'n_cycles': 10,
    },
}

DEFAULT_PARAMS = {
    'env_id': 'FetchReach-v1', # Try HalfCheetah-v2 for plain DDPG, FetchReach-v1 for HER
    'do_evaluation': True,
    'render_eval': False,
    'render_training': False,
    'seed': 42,
    'train_policy_fn': 'epsilon_greedy_noisy_explore',
    'eval_policy_fn': 'greedy_exploit',
    'agent_roles': 'exploit, explore',  # choices are 'explore, explore', 'exploit', and 'explore'
    'memory_type': 'replay_buffer',  # choices are 'replay_buffer' or 'ring_buffer'. 'ring_buffer' can't be used with HER.
    'separate_explore_ring_buffer': False,
    'heatmaps': False,  # generate heatmaps if using a gym-boxpush or FetchStack environment
    'boxpush_heatmaps': False,  # old argument, doesnt do anything, remaining to not break old scripts
    'map_dynamics_loss': False,

    # networks
    'exploit_layers': 3,  # number of layers in the critic/actor networks
    'exploit_hidden': 256,  # number of neurons in each hidden layers
    'explore_layers': 3,
    'explore_hidden': 256,
    'exploit_Q_lr': 0.001,  # critic learning rate
    'exploit_pi_lr': 0.001,  # actor learning rate
    'exploit_critic_l2_reg': 0,
    'explore_Q_lr': 0.001,  # critic learning rate
    'explore_pi_lr': 0.001,  # actor learning rate
    'explore_critic_l2_reg': 1e-2,
    'dynamics_layers': 3,
    'dynamics_hidden': 256,
    'dynamics_lr': 0.007, # dynamics module learning rate
    'exploit_polyak_tau': 0.001,  # polyak averaging coefficient (target_net = (1 - tau) * target_net + tau * main_net)
    'explore_polyak_tau': 0.05,  # polyak averaging coefficient (target_net = (1 - tau) * target_net + tau * main_net)
    'exploit_use_layer_norm': False,  # User layer normalization in actor critic networks
    'explore_use_layer_norm': True,  # User layer normalization in actor critic networks
    'exploit_gamma': 'auto',  # 'auto' or floating point number. If auto, gamma is 1 - 1/episode_time_horizon
    'explore_gamma': 'auto',  # 'auto' or floating point number. If auto, gamma is 1 - 1/episode_time_horizon
    'episode_time_horizon': 'auto',  # 'auto' or int. If 'auto' T is inferred from env._max_episode_steps

    # training
    'buffer_size': int(1E6),  # for experience replay
    'n_epochs': 25,
    'n_cycles': 50,  # per epoch
    'n_batches': 40,  # training batches per cycle
    'batch_size': 1024,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'rollout_batches_per_cycle': 8,
    'rollout_batch_size': 1,  # number of per mpi thread
    'n_test_rollouts': 50,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts

    # exploration
    'exploit_noise_type': 'normal_0.04',  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    'explore_noise_type': 'adaptive-param_0.1, normal_0.04',  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    'mix_extrinsic_intrinsic_objectives_for_explore': '0.5,0.5',
    'intrinsic_motivation_method': 'forward_dynamics',  # choices are 'forward_dynamics' or 'random_network_distillation'
    'num_model_groups': 1,

    # normalization
    'exploit_normalize_returns': True,
    'exploit_popart': True,
    'explore_normalize_returns': True,
    'explore_popart': True,
    'agents_normalize_observations': True,
    'agents_normalize_goals': True,
    'dynamics_normalize_observations': True,


    # HER
    'use_her': True,
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    'sub_goal_divisions': 'none',

    # Save and Restore
    'save_at_score': .98,  # success rate for HER, mean reward per episode for DDPG
    'stop_at_score': 'none',  # success rate for HER, mean reward per episode for DDPG
    'save_checkpoints_at': 'none',
    'restore_from_ckpt': 'none',
    'do_demo_only': False,
    'demo_video_recording_name': 'none',

    # GPU Usage Overrides
    'split_gpu_usage_among_device_nums': 'none'  # '[0, 1]' (list of gpu device nums) or 'none'
}

CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):

    env_id = kwargs['env_id']

    def make_env():
        return gym.make(env_id)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    kwargs['T'] = kwargs['episode_time_horizon']
    del kwargs['episode_time_horizon']
    if kwargs['T'] == 'auto':
        assert hasattr(tmp_env, '_max_episode_steps')
        kwargs['T'] = tmp_env._max_episode_steps
    else:
        kwargs['T'] = int(kwargs['T'])
    tmp_env.reset()

    if kwargs['use_her'] is False:
        # If HER is disabled, disable other HER related params.
        kwargs['replay_strategy'] = 'none'
        kwargs['replay_k'] = 0

    if 'BoxPush' not in kwargs['env_id'] and 'FetchStack' not in kwargs['env_id']:
        kwargs['heatmaps'] = False

    for gamma_key in ['exploit_gamma', 'explore_gamma']:
        kwargs[gamma_key] = 1. - 1. / kwargs['T'] if kwargs[gamma_key] == 'auto' else float(kwargs[gamma_key])

    if kwargs['map_dynamics_loss'] and 'BoxPush' in kwargs['env_id'] and 'explore' in kwargs['agent_roles']:
        kwargs['dynamics_loss_mapper'] = DynamicsLossMapper(
                working_dir=os.path.join(logger.get_dir(), 'dynamics_loss'),
                sample_env=cached_make_env(kwargs['make_env'])
            )
    else:
        kwargs['dynamics_loss_mapper'] = None

    for network in ['exploit', 'explore']:
        # Parse noise_type
        action_noise = None
        param_noise = None
        nb_actions = tmp_env.action_space.shape[-1]
        for current_noise_type in kwargs[network+'_noise_type'].split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                            sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
        kwargs[network+'_action_noise'] = action_noise
        kwargs[network+'_param_noise'] = param_noise
        del(kwargs[network+'_noise_type'])

    kwargs['train_rollout_params'] = {
        'compute_Q': False,
        'render': kwargs['render_training']
    }

    kwargs['eval_rollout_params'] = {
        'compute_Q': True,
        'render': kwargs['render_eval']
    }

    if kwargs['mix_extrinsic_intrinsic_objectives_for_explore'] == 'none':
        kwargs['mix_extrinsic_intrinsic_objectives_for_explore'] = None
    else:
        weights_string = kwargs['mix_extrinsic_intrinsic_objectives_for_explore']
        kwargs['mix_extrinsic_intrinsic_objectives_for_explore'] = [float(w) for w in weights_string.split(',')]
        assert len(kwargs['mix_extrinsic_intrinsic_objectives_for_explore']) == 2

    if kwargs['restore_from_ckpt'] == 'none':
        kwargs['restore_from_ckpt'] = None

    if kwargs['stop_at_score'] == 'none':
        kwargs['stop_at_score'] = None
    else:
        kwargs['stop_at_score'] = float(kwargs['stop_at_score'])

    if kwargs['sub_goal_divisions'] == 'none':
        kwargs['sub_goal_divisions'] = None
    else:
        sub_goal_string = kwargs['sub_goal_divisions']
        sub_goal_divisions = ast.literal_eval(sub_goal_string)

        assert type(sub_goal_divisions) == list
        for list_elem in sub_goal_divisions:
            assert type(list_elem) == list
            for index in list_elem:
                assert type(index) == int

        kwargs['sub_goal_divisions'] = sub_goal_divisions

    if kwargs['split_gpu_usage_among_device_nums'] == 'none':
        kwargs['split_gpu_usage_among_device_nums'] = None
    else:
        gpu_string = kwargs['split_gpu_usage_among_device_nums']
        gpu_nums = ast.literal_eval(gpu_string)
        assert len(gpu_nums) >= 1
        for gpu_num in gpu_nums:
            assert type(gpu_num) == int
        kwargs['split_gpu_usage_among_device_nums'] = gpu_nums

    original_COMM_WORLD_rank = MPI.COMM_WORLD.Get_rank()
    kwargs['explore_comm'] = MPI.COMM_WORLD.Split(color=original_COMM_WORLD_rank % kwargs['num_model_groups'],
                                                  key=original_COMM_WORLD_rank)

    if kwargs['save_checkpoints_at'] == 'none':
        kwargs['save_checkpoints_at'] = None
    else:
        save_checkpoints_list = ast.literal_eval(kwargs['save_checkpoints_at'])
        assert type(save_checkpoints_list) == list
        for i in range(len(save_checkpoints_list)):
            save_checkpoints_list[i] = float(save_checkpoints_list[i])
        kwargs['save_checkpoints_at'] = save_checkpoints_list

    if kwargs["demo_video_recording_name"] == 'none':
        kwargs["demo_video_recording_name"] = None
    else:
        assert type(kwargs["demo_video_recording_name"]) == str

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    obs = env.reset()

    if params['sub_goal_divisions'] is not None:
        assert len(obs['desired_goal']) == sum([len(divisions) for divisions in params['sub_goal_divisions']]) + 3
        assert np.array_equal(
            np.sort(np.concatenate(params['sub_goal_divisions'])),
            np.arange(0, len(obs['desired_goal'][:-3]))
        )

    her_params = {}

    if params['use_her']:

        def reward_fun(ag_1, g, info):  # vectorized
            batch_size = np.shape(g)[0]
            rewards = env.compute_reward(achieved_goal=ag_1, desired_goal=g, info=info)
            return np.resize(rewards, new_shape=(batch_size, 1))

        # Prepare configuration for HER.
        her_params['reward_fun'] = reward_fun

    else:
        her_params['reward_fun'] = None

    for name in ['replay_strategy', 'replay_k', 'sub_goal_divisions']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


# def simple_goal_subtract(a, b):
#     assert a.shape == b.shape
#     return a - b


# def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
#     sample_her_transitions = configure_her(params)
#     # Extract relevant parameters.
#     gamma = params['gamma']
#     rollout_batch_size = params['rollout_batch_size']
#     ddpg_params = params['ddpg_params']
#
#     input_dims = dims.copy()
#
#     # DDPG agent
#     env = cached_make_env(params['make_env'])
#     env.reset()
#     ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
#                         'T': params['T'],
#                         'clip_pos_returns': False,  # clip positive returns
#                         'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
#                         'rollout_batch_size': rollout_batch_size,
#                         'subtract_goals': simple_goal_subtract,
#                         'sample_transitions': sample_her_transitions,
#                         'gamma': gamma,
#                         })
#     ddpg_params['info'] = {
#         'env_name': params['env_name'],
#     }
#     policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
#     return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    assert len(env.action_space.shape) == 1

    if params['use_her']:
        assert len(obs['observation'].shape) == 1
        assert len(obs['desired_goal'].shape) == 1
        dims = {
            'o': obs['observation'].shape[0],
            'u': env.action_space.shape[0],
            'g': obs['desired_goal'].shape[0],
        }
    else:
        assert len(obs.shape) == 1
        dims = {
            'o': obs.shape[0],
            'u': env.action_space.shape[0],
        }

    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]

    return dims


def get_convert_arg_to_type_fn(arg_type):

    if arg_type == bool:
        def fn(value):
            if value in ['None', 'none']:
                return None
            if value in ['True', 'true', 't', '1']:
                return True
            elif value in ['False', 'false', 'f', '0']:
                return False
            else:
                raise ValueError("Argument must either be the string, \'True\' or \'False\'")
        return fn

    elif arg_type == int:
        def fn(value):
            if value in ['None', 'none']:
                return None
            return int(float(value))
        return fn
    elif arg_type == str:
        return lambda arg: arg
    else:
        def fn(value):
            if value in ['None', 'none']:
                return None
            return arg_type(value)
        return fn


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


def configure_replay_buffer(params):
    logger.info('Using Replay Buffer')

    sample_transitions = configure_her(params)
    input_dims = configure_dims(params)
    input_shapes = dims_to_shapes(input_dims)

    buffer_shapes = {key: (params['T'] if key != 'o' else params['T'] + 1, *input_shapes[key])
                     for key, val in input_shapes.items()}

    if params['use_her']:
        buffer_shapes['g'] = (buffer_shapes['g'][0], input_dims['g'])
        buffer_shapes['ag'] = (params['T'] + 1, input_dims['g'])
    else:
        buffer_shapes['r'] = (params['T'], 1)
        buffer_shapes['t'] = (params['T'], 1)

    buffer_size = (params['buffer_size'] // params['rollout_batch_size']) * params['rollout_batch_size']
    return ReplayBuffer(buffer_shapes, buffer_size, params['T'], sample_transitions, params['use_her'])


def configure_memory(params):
    memory_type = params['memory_type']
    if memory_type == 'replay_buffer':
        return configure_replay_buffer(params)
    elif memory_type == 'ring_buffer':
        raise NotImplementedError
    else:
        raise ValueError('memory_type must be \'replay_buffer\'.')


def configure_ddpg_agent(sess, role, memory, input_dims, external_critic_fn, params):
    input_shapes = dims_to_shapes(input_dims)
    observation_shape = input_shapes['o']
    goal_shape = input_shapes['g'] if params['use_her'] else None
    action_shape = input_shapes['u']
    action_dim = input_dims['u']

    if role == 'exploit':
        comm = MPI.COMM_WORLD
        use_goals = True if params['use_her'] else False
        use_intrinsic_reward = False
        dynamics_loss_mapper = None
        mix_external_critic_with_internal = None
        external_critic_fn = None

    elif role == 'explore':
        comm = params['explore_comm']
        assert comm != MPI.COMM_WORLD
        use_intrinsic_reward = True
        dynamics_loss_mapper = params['dynamics_loss_mapper']
        mix_external_critic_with_internal = params['mix_extrinsic_intrinsic_objectives_for_explore']
        if mix_external_critic_with_internal is not None:
            assert len(mix_external_critic_with_internal) == 2
            assert external_critic_fn is not None
            use_goals = True if params['use_her'] else False
        else:
            use_goals = False
            external_critic_fn = None

    else:
        raise ValueError('role must either be \'exploit\' or \'explore\'.')

    agent = DDPG(
        sess=sess,
        scope=role + '_ddpg', layer_norm=[role + '_use_layer_norm'], nb_actions=action_dim, memory=memory,
        observation_shape=observation_shape, action_shape=action_shape, goal_shape=goal_shape,
        param_noise=params[role + '_param_noise'], action_noise=params[role + '_action_noise'],
        gamma=params[role + '_gamma'], tau=params[role + '_polyak_tau'],
        normalize_returns=params[role + '_normalize_returns'], enable_popart=params[role + '_popart'],
        normalize_observations=params['agents_normalize_observations'],
        normalize_goals=params['agents_normalize_goals'], batch_size=params['batch_size'], observation_range=(-5., 5.),
        goal_range=(-200, 200), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=params[role + '_critic_l2_reg'], actor_lr=params[role + '_pi_lr'],
        critic_lr=params[role + '_Q_lr'], clip_norm=None, reward_scale=1., use_intrinsic_reward=use_intrinsic_reward,
        use_goals=use_goals, agent_hidden_layer_sizes=[params[role + '_hidden']] * params[role + '_layers'],
        dynamics_hidden=params['dynamics_hidden'], dynamics_layers=params['dynamics_layers'],
        dynamics_normalize_observations=params['dynamics_normalize_observations'],
        dynamics_loss_mapper=dynamics_loss_mapper, mix_external_critic_with_internal=mix_external_critic_with_internal,
        external_critic_fn=external_critic_fn, intrinsic_motivation_method=params['intrinsic_motivation_method'],
        comm=comm
    )
    logger.info('Using ' + role + ' agent.')
    # logger.info('Using ' + role + ' agent with the following configuration:')
    # logger.info(str(agent.__dict__.items()))

    return agent


def create_agents(sess, memory, input_dims, params):
    agent_roles = params['agent_roles'].replace(' ', '').split(',')
    agents = OrderedDict()

    if 'exploit' in agent_roles:
        role = 'exploit'
        agent = configure_ddpg_agent(sess=sess, role=role, memory=memory, input_dims=input_dims, params=params,
                                     external_critic_fn=None)
        agent.initialize()
        agent.reset()
        agents[role] = agent
        exploit_critic_fn = agent.critic_with_actor_fn
    else:
        exploit_critic_fn = None

    if 'explore' in agent_roles:
        role = 'explore'
        agent = configure_ddpg_agent(sess=sess, role=role, memory=memory, input_dims=input_dims, params=params,
                                     external_critic_fn=exploit_critic_fn)
        agent.initialize()
        agent.reset()
        agents[role] = agent

    return agents


def configure_rollout_worker(role, policy_fn, agents, dims, seed, logger, params):
    if role == 'train':
        rollout_key = 'train_rollout_params'
    elif role == 'eval':
        rollout_key = 'eval_rollout_params'
    else:
        raise ValueError('role must either be \'exploit\' or \'explore\'.')

    rollout_worker = RolloutWorker(
        policy_fn=policy_fn, agents=agents, dims=dims, logger=logger, make_env=params['make_env'], T=params['T'],
        use_her=params['use_her'], rollout_batch_size=params['rollout_batch_size'],
        compute_Q=params[rollout_key]['compute_Q'], render=params[rollout_key]['render'], history_len=100,
    )
    rollout_worker.seed(seed)
    return rollout_worker


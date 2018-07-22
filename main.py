from collections import namedtuple

import gym
import torch
import torch.nn as nn
import envs.treeqn.push
from agents.iterative_agents.iterative_agent import IterativeAgent
from agents.nstep_dqn_agent import NStepSynchronousDQNAgent
from losses.auxiliary_losses.probable_action_loss import ProbableActionLoss
from losses.auxiliary_losses.reward_loss import RewardLoss
from losses.auxiliary_losses.samples_action_loss import SamplesActionLoss
from losses.auxiliary_losses.tree_nstep_reward_loss import TreeNStepRewardLoss
from models.classic_control.simple_cartpole_model import SimpleCartPoleModel
#from envs.atari.atari_wrapper import wrap_deepmind
from agents.dqn_agent import DQNAgent
from models.iterative.feature_models.push_model import PushModel
from models.iterative.value_models.q_model import QModel
from models.maze.fc_maze import FCMaze
from models.treeqn.push_dqn_model import PushDQNModel
from models.treeqn.push_tree_model import PushTreeModel
from losses.td_losses.q_loss import QLoss
from envs.wrappers.maze_wrappers import CorrectActionWrapper, MaxStepWrapper
from utils.scheduler.linear_scheduler import LinearScheduler
from utils.vec_env.subproc_vec_env import SubprocVecEnv
from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import WaterMazeGenerator


def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        # env = PyTorchImageWrapper(env)
        #env = wrap_deepmind(env)
        # print('max_steps:', env._max_episode_steps)
        env.seed(seed)
        return env

    return _f

def prep_env(env):
    def _f():
        return env
    return _f

cuda = False
#experiment = 'PushDQN'
#experiment = 'PushNStepSyncDQN'
#experiment = 'CartPoleDQN'
#experiment = 'CartPoleNStepSynchronousDQN'
#experiment = 'PushIterativeDQN'
experiment='TMaze'

if cuda:
    device = 'cuda'
else:
    device = 'cpu'

auxiliary_env_info = namedtuple('auxiliary_env_info', 'names, types')

if experiment == 'CartPoleDQN':
    env = gym.make('CartPole-v0')
    agent = DQNAgent(experiment, SimpleCartPoleModel, [4, 2], None, n_episodes=50000, replay_buffer_size=100000, device=device,
                     epsilon_scheduler_use_steps=True, epsilon_scheduler=LinearScheduler(decay_steps=int(15e3)),
                     target_synchronize_steps=10000, grad_clamp=[-1, 1],
                     training_evaluation_frequency=100, td_losses=[QLoss()], auxiliary_env_info=
                     auxiliary_env_info(['actions', 'boxtion'], [torch.long, torch.long]))
    agent.learn(env, env)

elif experiment == 'CartPoleNStepSynchronousDQN':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    nproc = 8
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)  # target_sync = 10e4 * n_proc
    auxiliary_env_info = namedtuple('auxiliary_env_info', 'names, types')
    agent = NStepSynchronousDQNAgent(experiment, SimpleCartPoleModel, [4, 2], None, n_processes=nproc, device=device,
                                     target_synchronize_steps=10000, grad_clamp=[-1, 1], td_losses=[QLoss()],
                                     training_evaluation_frequency=10000)
    agent.learn(envs, env)

elif experiment == 'PushNStepSyncDQN':
    env_name = 'Push-v0'
    env = gym.make(env_name)
    #env = PyTorchImageWrapper(env)
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    # params
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = NStepSynchronousDQNAgent(experiment, PushTreeModel, [5, 4, 2], None, n_processes=nproc, device=device,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp=[-1, 1], training_evaluation_frequency=40000, criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler(decay_steps=5e4), td_losses=[QLoss()],
                                     auxiliary_losses=[TreeNStepRewardLoss(2, 5, nproc)], checkpoint_epsilon=False)
    agent.learn(envs, env)
    #agent._eval(env)

elif experiment == 'PushDQN':
    env_name = 'Push-v0'
    env = gym.make(env_name)

    #env = PyTorchImageWrapper(env)
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    # agent = DQNAgent(experiment, PushDQNModel, [5, 4], None, device=device, optimizer_parameters=optimizer_parameters,
    #                  target_synchronize_steps=40000, grad_clamp=[-1, 1], training_evaluation_frequency=2500,
    #                  epsilon_scheduler=LinearScheduler(decay_steps=4e6), epsilon_scheduler_use_steps=True,
    #                  criterion=nn.MSELoss)

    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    agent = NStepSynchronousDQNAgent(experiment, PushDQNModel, [5, 4], None, n_processes=nproc, device=device,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp=[-1, 1], training_evaluation_frequency=40000, criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler(decay_steps=5e4), td_losses=[QLoss()],
                                     checkpoint_epsilon=True)
    agent.learn(envs, env)

elif experiment == 'PushIterativeDQN':
    experiment += '_both_target'
    env_name = 'Push-v0'
    env = gym.make(env_name)
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)
    feature_agent = NStepSynchronousDQNAgent(experiment+'_feature', PushModel, [5, 4], None, n_processes=nproc,
                                             device=device, optimizer_parameters=optimizer_parameters,
                                             target_synchronize_steps=40000,  grad_clamp=[-1, 1], td_losses=None,
                                             auxiliary_losses=[RewardLoss()],
                                             training_evaluation_frequency=10000, criterion=nn.MSELoss,
                                             epsilon_scheduler=LinearScheduler(decay_steps=5e4))
    q_agent = NStepSynchronousDQNAgent(experiment+'_value', QModel, [512, 512, 4], None, n_processes=nproc,
                                       device=device, optimizer_parameters=optimizer_parameters,
                                       target_synchronize_steps=40000, grad_clamp=[-1, 1],
                                       training_evaluation_frequency=10000, criterion=nn.MSELoss, td_losses=[QLoss()],
                                       epsilon_scheduler=LinearScheduler(decay_steps=5e4))
    agent = IterativeAgent(feature_agent, q_agent)
    agent.learn(envs, envs, env, env)

elif experiment == 'SeaquestNStepSyncDQN':
    env_name = 'SeaquestNoFrameskip-v4'
    env = wrap_deepmind(gym.make(env_name))
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    # params
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = NStepSynchronousDQNAgent(experiment, PushTreeModel, [5, 4, 2], None, n_processes=nproc, device=device,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp=[-1, 1], training_evaluation_frequency=2500, criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler())
    agent.learn(envs, env)

elif experiment == 'TMaze':
    maze = WaterMazeGenerator(7, 1)
    action_type = 'VonNeumann'
    env = MazeEnv(maze, action_type=action_type, render_trace=True, live_display=True)
    w_env = MaxStepWrapper(CorrectActionWrapper(env), 200)
    m_env = MaxStepWrapper(env, 100)
    # nproc = 8
    # envs = [prep_env(MazeEnv(maze, action_type=action_type, render_trace=False)) for seed in range(nproc)]
    # envs = SubprocVecEnv(envs)

    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = DQNAgent(experiment, FCMaze, [196, 4], None, n_episodes=50000, replay_buffer_size=10000, device=device,
                     epsilon_scheduler_use_steps=True, epsilon_scheduler=LinearScheduler(decay_steps=int(1e4)),
                     target_synchronize_steps=5000, grad_clamp=[-1, 1],
                     training_evaluation_frequency=1, td_losses=[QLoss()], log=True,
                     auxiliary_env_info=auxiliary_env_info(['actions'], [torch.long]), auxiliary_losses=[SamplesActionLoss()])
    agent.learn(w_env, m_env, n_eval_episodes=10)

elif experiment == 'NStepTMaze':
    # maze = WaterMazeGenerator(7, 1)
    # action_type = 'VonNeumann'
    # env = MazeEnv(maze, action_type=action_type, render_trace=True, live_display=True)
    # w_env = MaxStepWrapper(CorrectActionWrapper(env), 200)
    # m_env = MaxStepWrapper(env, 200)
    # nproc = 8
    # envs = [prep_env(MaxStepWrapper(CorrectActionWrapper(MazeEnv(maze, action_type=action_type, render_trace=False)), 200)) for seed in range(nproc)]
    # envs = SubprocVecEnv(envs)
    #
    # optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    # agent = NStepSynchronousDQNAgent(experiment, FCMaze, [400, 4], None, n_processes=nproc, device=device,
    #                                  target_synchronize_steps=1000, grad_clamp=[-1, 1], td_losses=[QLoss()],
    #                                  training_evaluation_frequency=2000, auxiliary_losses=[SamplesActionLoss()],
    #                                  auxiliary_env_info=auxiliary_env_info(['actions'], [torch.long]), log=True)
    # agent.learn(w_env, m_env, n_eval_episodes=10)
    pass

elif experiment == 'debug':
    env_name = 'Push-v0'
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    model = PushTreeModel(5, 4)
    for param in model.parameters():
        param.data = param.data - param.data + 1.
        pass
    s = envs.reset()
    s = torch.tensor(s, dtype=torch.float, device=device)
    output = model(s)
    print('output:', output)

import torch

from agents.nstep_dqn_agent import NStepSynchronousDQNAgent
from auxiliary_losses.tree_nstep_reward_loss import TreeNStepRewardLoss
from models.classic_control.simple_cartpole_model import SimpleCartPoleModel
import gym
import envs.treeqn.push
#from envs.atari.atari_wrapper import wrap_deepmind
from agents.dqn_agent import DQNAgent
from models.treeqn.push_model import PushModel
from auxiliary_losses.tree_reward_loss import TreeRewardLoss
from utils.scheduler.linear_scheduler import LinearScheduler
from utils.scheduler.decay_scheduler import DecayScheduler
from utils.vec_env.subproc_vec_env import SubprocVecEnv
import torch.nn as nn


def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        #env = wrap_deepmind(env)
        # print('max_steps:', env._max_episode_steps)
        env.seed(seed)
        return env

    return _f


cuda = False
# experiment = 'PushNStepSyncDQN'
experiment = 'CartPoleDQN'
# experiment = 'CartPoleNStepSynchronousDQN'

if cuda:
    device = 'cuda'
else:
    device = 'cpu'

if experiment == 'CartPoleDQN':
    env = gym.make('CartPole-v0')
    agent = DQNAgent(experiment, SimpleCartPoleModel, [4, 2], None, n_episodes=50000, replay_buffer_size=100000, device=device,
                     epsilon_scheduler_use_steps=True, target_synchronize_steps=10000, grad_clamp=[-1, 1],
                     training_evaluation_frequency=100)
    agent.learn(env, env)

elif experiment == 'CartPoleNStepSynchronousDQN':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    nproc = 8
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)  # target_sync = 10e4 * n_proc
    agent = NStepSynchronousDQNAgent(experiment, SimpleCartPoleModel, [4, 2], None, n_processes=nproc, device=device,
                                     target_synchronize_steps=10000, grad_clamp=[-1, 1], training_evaluation_frequency=10000)
    agent.learn(envs, env)

elif experiment == 'PushNStepSyncDQN':
    env_name = 'Push-v0'
    env = gym.make(env_name)
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    # params
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = NStepSynchronousDQNAgent(experiment, PushModel, [5, 4, 2], None, n_processes=nproc, device=device,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp=[-1, 1], training_evaluation_frequency=2500, criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler(), auxiliary_losses=[TreeNStepRewardLoss(2, 5,
                                                                                                                nproc)])
    agent.learn(envs, env)

elif experiment == 'PushDQN':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = DQNAgent(experiment, PushModel, [5, 4, 2], None, device=device, optimizer_parameters=optimizer_parameters,
                     target_synchronize_steps=40000, grad_clamp=[-1, 1], training_evaluation_frequency=2500,
                     epsilon_scheduler=LinearScheduler(), epsilon_scheduler_use_steps=True, criterion=nn.MSELoss
                     )
    agent.learn(envs, env)

elif experiment == 'SeaquestNStepSyncDQN':
    env_name = 'SeaquestNoFrameskip-v4'
    env = wrap_deepmind(gym.make(env_name))
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    # params
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = NStepSynchronousDQNAgent(experiment, PushModel, [5, 4, 2], None, n_processes=nproc, device=device,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp=[-1, 1], training_evaluation_frequency=2500, criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler())
    agent.learn(envs, env)
elif experiment == 'debug':
    env_name = 'Push-v0'
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    model = PushModel(5, 4)
    for param in model.parameters():
        param.data = param.data - param.data + 1.
        pass
    s = envs.reset()
    s = torch.tensor(s, dtype=torch.float, device=device)
    output = model(s)
    print('output:', output)

import gym
import envs.treeqn.push
import torch
import torch.nn as nn

from agents.iterative_agents.iterative_agent import IterativeAgent
from agents.nstep_dqn_agent import NStepSynchronousDQNAgent
from losses.auxiliary_losses.reward_loss import RewardLoss
from losses.auxiliary_losses.tree_nstep_reward_loss import TreeNStepRewardLoss
from models.classic_control.simple_cartpole_model import SimpleCartPoleModel
#from envs.atari.atari_wrapper import wrap_deepmind
from agents.dqn_agent import DQNAgent
from models.iterative.feature_models.push_model import PushModel
from models.iterative.value_models.q_model import QModel
from models.treeqn.push_dqn_model import PushDQNModel
from models.treeqn.push_tree_model import PushTreeModel
from losses.td_losses.q_loss import QLoss
from utils.scheduler.linear_scheduler import LinearScheduler
from utils.vec_env.subproc_vec_env import SubprocVecEnv


def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        # env = PyTorchImageWrapper(env)
        #env = wrap_deepmind(env)
        # print('max_steps:', env._max_episode_steps)
        env.seed(seed)
        return env

    return _f


cuda = True
#experiment = 'PushDQN'
experiment = 'PushNStepSyncDQN'
#experiment = 'CartPoleDQN'
#experiment = 'CartPoleNStepSynchronousDQN'
#experiment = 'PushIterativeDQN'

if cuda:
    device = 'cuda'
else:
    device = 'cpu'

if experiment == 'CartPoleDQN':
    env = gym.make('CartPole-v0')
    agent = DQNAgent(experiment, SimpleCartPoleModel, [4, 2], None, n_episodes=50000, replay_buffer_size=100000, device=device,
                     epsilon_scheduler_use_steps=True, epsilon_scheduler=LinearScheduler(decay_steps=int(15e3)),
                     target_synchronize_steps=10000, grad_clamp=[-1, 1],
                     training_evaluation_frequency=100, td_losses=[QLoss()])
    agent.learn(env, env)

elif experiment == 'CartPoleNStepSynchronousDQN':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    nproc = 8
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)  # target_sync = 10e4 * n_proc
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
                                     epsilon_scheduler=LinearScheduler(decay_steps=1e5), td_losses=[QLoss()],
                                     auxiliary_losses=[TreeNStepRewardLoss(2, 5, nproc)], checkpoint_epsilon=True)
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

import pickle
import time
import numpy as np
import random
import numpy
import os
from collections import namedtuple

import gym
import torch
import torch.nn as nn
from torch import optim

import envs.treeqn.push
from agents.checkpoint.nstep_dqn_checkpoint_agent import NStepDQNCheckpointAgent
from agents.iterative_agents.iterative_agent import IterativeAgent
from agents.nstep_dqn_agent import NStepSynchronousDQNAgent
from agents.tabular.q_learner import QLearner
from envs.wrappers.atari_wrappers_new import wrap_deepmind2, make_atari
from envs.wrappers.frame_stack_wrappers import MultiProcessFrameStackWrapper, FrameStackWrapper
from envs.wrappers.pytorch_image_wrapper import PyTorchImageWrapper
from losses.auxiliary_losses.probable_action_loss import ProbableActionLoss
from losses.auxiliary_losses.reward_loss import RewardLoss
from losses.auxiliary_losses.samples_action_loss import SamplesActionLoss
from losses.auxiliary_losses.tree_nstep_reward_loss import TreeNStepRewardLoss
from models.atari.atari_model import AtariModel
from models.classic_control.simple_cartpole_model import SimpleCartPoleModel
from envs.wrappers.atari_wrappers import wrap_deepmind
from agents.dqn_agent import DQNAgent
from models.iterative.feature_models.push_model import PushModel
from models.iterative.value_models.q_model import QModel
from models.maze.fc_maze import FCMaze
from models.treeqn.atari_model import AtariTreeModel
from models.treeqn.push_fc_model import PushFCModel
from models.treeqn.push_tree_model import PushTreeModel
from losses.td_losses.q_loss import QLoss
# from envs.wrappers.maze_wrappers import CorrectActionWrapper, MaxStepWrapper, FixedRandomEnvsWrapper
from utils.scheduler.constant_scheduler import ConstantScheduler
from utils.scheduler.linear_scheduler import LinearScheduler
from utils.transforms.input_transforms.atari_lazy_numpy_image_transform import AtariLazyNumpyImageTransform
from utils.vec_env.subproc_vec_env import SubprocVecEnv
# from gym_maze.envs import MazeEnv
# from gym_maze.envs.generators import WaterMazeGenerator
import socket
from datetime import datetime


def make_env(env_id, seed, image_wrap=True, wrap=False, skip=10):
    def _f():
        env = gym.make(env_id)
        if wrap:
            # env = wrap_deepmind(env, episode_life=True, clip_rewards=True, skip=skip)
            env = wrap_deepmind2(env, episode_life=True, clip_rewards=True, frame_stack=10)
        # print('max_steps:', env._max_episode_steps)
        # env.seed(seed)
        if image_wrap:
            env = PyTorchImageWrapper(env)
        return env

    return _f

def make_atari_env(env_id):
    def _f():
        env = make_atari(env_id, skip=10)
        env = wrap_deepmind2(env, frame_stack=True)
        return env
    return _f


atari_image_transform = AtariLazyNumpyImageTransform()


def prep_env(env):
    def _f():
        return env

    return _f

cuda = True
# cuda = False
checkpoint = True
# checkpoint = False
# experiment = 'PushDQN'
# experiment = 'PushNStepSyncDQN'
# experiment = 'CartPoleDQN'
# experiment = 'CartPoleNStepSynchronousDQN'
# experiment = 'PushIterativeDQN'
# experiment = 'TMaze'
# experiment = 'SeaquestNStep'
# experiment = 'PongNStep'
# experiment = 'PongDQN'
# experiment = 'test-pong'
# experiment = 'frozenLake'
# experiment = 'FrostbiteNStep'
# experiment = 'EnduroNStep'
experiment = 'atariNStep'
# experiment = 'debug'


def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)


if cuda:
    device = 'cuda'
else:
    device = 'cpu'
print('running', experiment)
auxiliary_env_info = namedtuple('auxiliary_env_info', 'names, types')

if checkpoint:
    prefix = 'Checkpoint'
else:
    prefix = 'Pure'

current_time = datetime.now().strftime('%b%d_%H-%M-%S')


if experiment == 'CartPoleDQN':
    env = gym.make('CartPole-v0')
    decay_steps = int(1e4)
    log_dir = os.path.join('runs', experiment, prefix, str(decay_steps),
                           '_' + current_time + '_' + socket.gethostname())
    optimizer_parameters = {'lr': 1e-3, 'alpha': 0.99, 'eps': 1e-5}
    agent = DQNAgent(SimpleCartPoleModel, [4, 2], None, n_episodes=50000, replay_buffer_size=50000, device=device,
                     epsilon_scheduler_use_steps=True, epsilon_scheduler=LinearScheduler(decay_steps=decay_steps,
                                                                                         final_epsilon=0.02),
                     target_synchronize_steps=500, grad_clamp='norm', grad_clamp_parameters=[10],
                     training_evaluation_frequency=100, td_losses=[QLoss()], log=True, log_dir=log_dir,
                     checkpoint_epsilon=checkpoint, checkpoint_epsilon_scheduler_template=LinearScheduler(decay_steps=int(15e1),
                                                                                                    final_epsilon=0.02),
                     checkpoint_epsilon_frequency=5,
                     save_checkpoint=False)

    # agent = DQNAgent(SimpleCartPoleModel, [4, 2], None, n_episodes=50000, replay_buffer_size=50000, device=device,
    #                  epsilon_scheduler_use_steps=True, epsilon_scheduler=LinearScheduler(decay_steps=int(1e4), lower_bound=0.02),
    #                  target_synchronize_steps=500, grad_clamp=[-1, 1],
    #                  training_evaluation_frequency=100, td_losses=[QLoss()], log=False,
    #                  replay_buffer_min_experience=1000, gamma=1.)

    agent.learn(env, env)

elif experiment == 'CartPoleNStepSynchronousDQN':
    env_name = 'CartPole-v0'
    decay_steps = int(50e3)
    log_dir = os.path.join('runs', experiment, prefix, str(decay_steps),
                           '_' + current_time + '_' + socket.gethostname())

    env = gym.make(env_name)
    nproc = 8
    envs = [make_env(env_name, seed, image_wrap=False) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)  # target_sync = 10e4 * n_proc
    auxiliary_env_info = namedtuple('auxiliary_env_info', 'names, types')
    agent = NStepSynchronousDQNAgent(SimpleCartPoleModel, [4, 2], None, n_processes=nproc, device=device,
                                     target_synchronize_steps=10000, grad_clamp='2', grad_clamp_parameters=[-1, 1],
                                     td_losses=[QLoss()], epsilon_scheduler=LinearScheduler(decay_steps=decay_steps,
                                                                                            final_epsilon=0.05),
                                     training_evaluation_frequency=10000, log=True, log_dir=log_dir, save_checkpoint=True)
    # path = os.path.join('temp_runs', experiment, prefix, str(int(5e4)), '_Aug09_23-43-11_agent22-ml')
    # agent.load(path)
    agent.learn(envs, env)

elif experiment == 'PushNStepSyncDQN':

    # log dir name
    nproc = 16
    nstep = 5
    warmup = True
    if checkpoint:
        decay_steps = int(1e4)  # TODO always look here.....
    else:
        decay_steps = int(4e6)
    warmup_steps = int(2e6)  # actually 2x warmup steps
    if warmup:
        prefix = os.path.join(prefix, 'warmup', str(2*warmup_steps))
    log_dir = os.path.join('runs', experiment, prefix, str(decay_steps),
                           '_' + current_time + '_' + socket.gethostname())
    if not checkpoint:
        decay_steps = decay_steps // (nproc * nstep)
        checkpoint_freq = None
    else:
        checkpoint_freq = int(1e4)


    env_name = 'Push-v0'
    env = gym.make(env_name)
    env = PyTorchImageWrapper(env)
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    # params
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = NStepSynchronousDQNAgent(PushTreeModel, [5, 4, 2], None, n_processes=nproc, device=device, n_step=nstep,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp='value',
                                     grad_clamp_parameters=[-1, 1], training_evaluation_frequency=40000, criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler(decay_steps=warmup_steps//(nstep*nproc)),
                                     checkpoint_warmup_steps=2 * warmup_steps,
                                     checkpoint_epsilon_scheduler_template=LinearScheduler(decay_steps=decay_steps),
                                     td_losses=[QLoss()], auxiliary_losses=[TreeNStepRewardLoss(2, 5, nproc)],
                                     checkpoint_epsilon=checkpoint, checkpoint_epsilon_frequency=checkpoint_freq,
                                     log=True, log_dir=log_dir)
    agent.learn(envs, env)
    # agent._eval(env)

elif experiment == 'PushDQN':
    env_name = 'Push-v0'
    env = gym.make(env_name)

    # env = PyTorchImageWrapper(env)
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    # agent = DQNAgent(experiment, PushDQNModel, [5, 4], None, device=device, optimizer_parameters=optimizer_parameters,
    #                  target_synchronize_steps=40000, grad_clamp=[-1, 1], training_evaluation_frequency=2500,
    #                  epsilon_scheduler=LinearScheduler(decay_steps=4e6), epsilon_scheduler_use_steps=True,
    #                  criterion=nn.MSELoss)
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)

    agent = NStepSynchronousDQNAgent(PushFCModel, [5, 4], None, n_processes=nproc, device=device,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp=[-1, 1], training_evaluation_frequency=40000, criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler(decay_steps=decay_steps), td_losses=[QLoss()],
                                     checkpoint_epsilon=checkpoint)
    agent.learn(envs, env)

elif experiment == 'PushIterativeDQN':
    experiment += '_both_target'
    env_name = 'Push-v0'
    env = gym.make(env_name)
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)
    feature_agent = NStepSynchronousDQNAgent(experiment + '_feature', PushModel, [5, 4], None, n_processes=nproc,
                                             device=device, optimizer_parameters=optimizer_parameters,
                                             target_synchronize_steps=40000, grad_clamp=[-1, 1], td_losses=None,
                                             auxiliary_losses=[RewardLoss()],
                                             training_evaluation_frequency=10000, criterion=nn.MSELoss,
                                             epsilon_scheduler=LinearScheduler(decay_steps=5e4))
    q_agent = NStepSynchronousDQNAgent(experiment + '_value', QModel, [512, 512, 4], None, n_processes=nproc,
                                       device=device, optimizer_parameters=optimizer_parameters,
                                       target_synchronize_steps=40000, grad_clamp=[-1, 1],
                                       training_evaluation_frequency=10000, criterion=nn.MSELoss, td_losses=[QLoss()],
                                       epsilon_scheduler=LinearScheduler(decay_steps=5e4))
    agent = IterativeAgent(feature_agent, q_agent)
    agent.learn(envs, envs, env, env)

elif experiment == 'TMaze':
    action_type = 'VonNeumann'
    # env = MazeEnv(maze, action_type=action_type, render_trace=True, live_display=True)
    envs = []
    # ensure all envs are solvable
    for i in range(5):
        maze = WaterMazeGenerator(7, 1, obstacle_ratio=0.3)
        env = CorrectActionWrapper(MazeEnv(maze, action_type=action_type, render_trace=True, live_display=False),
                                   flush_cache=False)
        while not env.is_solvable():
            maze = WaterMazeGenerator(7, 1, obstacle_ratio=0.3)
            env = CorrectActionWrapper(MazeEnv(maze, action_type=action_type, render_trace=True, live_display=False),
                                   flush_cache=False)
        #env.render()
        envs.append(MaxStepWrapper(env, 100))
    print('starting experiments')
    w_env = FixedRandomEnvsWrapper(envs)
    # m_env = MaxStepWrapper(env, 100)
    # nproc = 8
    # envs = [prep_env(MazeEnv(maze, action_type=action_type, render_trace=False)) for seed in range(nproc)]
    # envs = SubprocVecEnv(envs)

    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = DQNAgent(experiment, FCMaze, [196, 4], None, n_episodes=50000, replay_buffer_size=50000, device=device,
                     epsilon_scheduler_use_steps=True, epsilon_scheduler=LinearScheduler(decay_steps=int(1e4)),
                     target_synchronize_steps=2500, grad_clamp=[-1, 1],
                     training_evaluation_frequency=10, td_losses=[QLoss()], log=True,
                     auxiliary_env_info=auxiliary_env_info(['actions'], [torch.long]), auxiliary_losses=
                     [SamplesActionLoss(n_samples=4)])
    agent.learn(w_env, w_env, n_eval_episodes=10)

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

elif experiment == 'debug-tree':
    env_name = 'Push-v0'
    nproc = 16
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    model = PushTreeModel(5, 4)
    for param in model.parameters():
        param.data = param.data - param.data + 0.5
        pass
    s = envs.reset()
    s = torch.tensor(s, dtype=torch.float, device=device)
    output = model(s)
    print('output:', output)

elif experiment == 'PongNStep':
    nproc = 16
    nstep = 5
    if checkpoint:
        decay_steps = int(25e2)  # TODO always look here.....
    else:
        decay_steps = int(4e6)
    log_dir = os.path.join('runs', experiment, prefix, str(decay_steps),
                           '_' + current_time + '_' + socket.gethostname())
    if not checkpoint:
        decay_steps = decay_steps // (nproc * nstep)
        checkpoint_freq = None
    else:
        checkpoint_freq = int(0.25 * 40000)

    env_name = 'PongNoFrameskip-v4'
    env = [make_env(env_name, 100, wrap=True, skip=10) for seed in range(1)]
    env = SubprocVecEnv(env)
    env = FrameStackWrapper(env, 1, n_stack=4)
    envs = [make_env(env_name, seed, wrap=True, skip=10) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)
    envs = MultiProcessFrameStackWrapper(envs, nproc, n_stack=4)

    # params
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = NStepSynchronousDQNAgent(AtariTreeModel, [4, env.action_space.n], None, n_processes=nproc, device=device,
                                     n_step=nstep,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp=[-1, 1], training_evaluation_frequency=40000, criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler(decay_steps=decay_steps),
                                     td_losses=[QLoss()], auxiliary_losses=[TreeNStepRewardLoss(2, 5, nproc)],
                                     checkpoint_epsilon=checkpoint, checkpoint_epsilon_frequency=checkpoint_freq,
                                     log=True, log_dir=log_dir)
    agent.learn(envs, env, n_eval_episodes=5)

elif experiment == 'PongDQN':
    if checkpoint:
        decay_steps = int(1e4)  # TODO always look here.....
    else:
        decay_steps = int(1e5)
    log_dir = os.path.join('runs', experiment, prefix, str(decay_steps),
                           '_' + current_time + '_' + socket.gethostname())
    if not checkpoint:
        checkpoint_freq = None
    else:
        checkpoint_freq = 5  # episode

    env_name = 'PongNoFrameskip-v4'
    env = [make_env(env_name, 100, wrap=True, skip=4) for seed in range(1)]
    env = SubprocVecEnv(env)
    env = FrameStackWrapper(env, 1, n_stack=4)

    # optimizer_parameters = {'lr': 0.0001, 'alpha': 0.99, 'eps': 1e-2, 'momentum': 0.95, }
    optimizer_parameters = {'lr': 0.0001}
    agent = DQNAgent(AtariModel, [4, env.action_space.n], None, device=device, optimizer=optim.Adam,
                     optimizer_parameters=optimizer_parameters,
                     target_synchronize_steps=1000, grad_clamp=[-1, 1], training_evaluation_frequency=30,
                     epsilon_scheduler=LinearScheduler(decay_steps=decay_steps, final_epsilon=0.02), epsilon_scheduler_use_steps=True,
                     replay_buffer_size=int(1e5), replay_buffer_min_experience=int(1e4), n_episodes=int(1e8),
                     criterion=nn.MSELoss, td_losses=[QLoss()], checkpoint_epsilon=checkpoint,
                     checkpoint_epsilon_frequency=checkpoint_freq, log=True, log_dir=log_dir)
    agent.learn(env, env, n_eval_episodes=3)


elif experiment == 'atariNStep':
    env_name = 'Qbert'
    print('running', env_name)

    experiment = env_name + 'NStep'
    # log dir name
    nproc = 16
    nstep = 5
    if checkpoint:
        decay_steps = int(5e4)  # TODO always look here...
    else:
        decay_steps = int(4e6)
    log_dir = os.path.join('runs', experiment, prefix, str(decay_steps),
                           '_' + current_time + '_' + socket.gethostname())
    if not checkpoint:
        decay_steps = decay_steps // (nproc * nstep)
        checkpoint_freq = None
    else:
        checkpoint_freq = int(25e4)

    env_name = env_name+'NoFrameskip-v4'
    # env = [make_env(env_name, 100, wrap=True, skip=10) for seed in range(1)]
    # env = SubprocVecEnv(env)
    # env = FrameStackWrapper(env, 1, n_stack=4)
    # envs = [make_env(env_name, seed, wrap=True, skip=10) for seed in range(nproc)]
    # envs = SubprocVecEnv(envs)
    # envs = MultiProcessFrameStackWrapper(envs, nproc, n_stack=4)
    env = make_atari(env_name, skip=10)
    env = wrap_deepmind2(env, episode_life=False, clip_rewards=False, frame_stack=True)
    envs = [make_atari_env(env_name) for _ in range(nproc)]
    envs = SubprocVecEnv(envs)

    # params
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    def constant_fn(*args):
        return 1.0
    lr_fn = constant_fn
    agent = NStepDQNCheckpointAgent(AtariTreeModel, [4, env.action_space.n], None, n_processes=nproc, device=device, n_step=nstep,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     lr_scheduler_fn=lr_fn, input_transforms=[atari_image_transform],
                                     grad_clamp='norm', grad_clamp_parameters=[5], training_evaluation_frequency=40000,
                                     criterion=nn.MSELoss, epsilon_scheduler=LinearScheduler(decay_steps=decay_steps),
                                     checkpoint_warmup_steps=0, checkpoint_epsilon_scheduler_template=LinearScheduler(decay_steps=decay_steps),
                                     td_losses=[QLoss(criterion=nn.MSELoss)], auxiliary_losses=[TreeNStepRewardLoss(2, 5, nproc)],
                                     checkpoint_epsilon_frequency=checkpoint_freq, log=True, log_dir=log_dir)
    agent.learn(envs, env, n_eval_episodes=5)

elif experiment == 'debug':
    env_name = 'AmidarNoFrameskip-v4'
    # env = [make_env(env_name, 100, wrap=True, skip=4) for seed in range(3)]
    # env = SubprocVecEnv(env)
    # env = MultiProcessFrameStackWrapper(env, 3, n_stack=4)
    env = gym.make(env_name)
    # env = wrap_deepmind2(env, frame_stack=4)
    env = wrap_deepmind(env, skip=10)
    # ones = torch.from_numpy(numpy.ones((1, 4, 84, 84))).float()

    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}


    def constant_fn(*args):
        return 1.0


    lr_fn = constant_fn
    nproc = 16
    nstep = 5
    decay_steps = int(4e6)
    log=True
    log_dir='/home/ml/svenka10//PycharmProjects/rl/runs/AmidarDebug'
    checkpoint_freq=None
    save_checkpoint = False
    agent = NStepSynchronousDQNAgent(AtariTreeModel, [4, env.action_space.n], None, n_processes=nproc, device=device,
                                     n_step=nstep,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     lr_scheduler_fn=lr_fn, input_transforms=[atari_image_transform],
                                     grad_clamp='norm', grad_clamp_parameters=[5], training_evaluation_frequency=40000,
                                     criterion=nn.MSELoss,
                                     epsilon_scheduler=LinearScheduler(decay_steps=decay_steps),
                                     td_losses=[QLoss(criterion=nn.MSELoss)],
                                     auxiliary_losses=[TreeNStepRewardLoss(2, 5, nproc)],
                                     checkpoint_epsilon=checkpoint, checkpoint_epsilon_frequency=checkpoint_freq,
                                     log=log, log_dir=log_dir, save_checkpoint=save_checkpoint)

    model = agent.model_learner
    for param in model.parameters():
        param.data = param.data * 0. + 0.5
    model = agent.model_target
    for param in model.parameters():
        param.data = param.data * 0. + 0.5
    for i in range(1, 1001):
        print('iteration', i)
        mb_data = torch.load('/home/ml/svenka10/debug_data/mb_data_' + str(i))
        next_states = mb_data[1].transpose(0, 3, 1, 2)
        next_states = next_states.reshape(nproc, nstep, 4, 84, 84).transpose(1, 0, 2, 3, 4)
        states = mb_data[0].reshape(nproc, nstep, 4, 84, 84).transpose(1, 0, 2, 3, 4)
        actions = mb_data[5].reshape(nproc, nstep, 1).transpose(1, 0, 2)
        rewards = mb_data[3].reshape(nproc, nstep, 1).transpose(1, 0, 2)
        dones = mb_data[7].reshape(nproc, nstep, 1).transpose(1, 0, 2)
        batch = agent._get_batch2(states, actions, next_states, rewards, dones)
        temp = batch[3][0].view(-1, 1).cpu().numpy().reshape(-1, 16).swapaxes(1, 0).flatten()
        print('diff:', numpy.sum(temp - mb_data[2].flatten()))
        print('indices:', np.nonzero(temp - mb_data[2].flatten()))
        _l = len(list(model.parameters()))
        _sum = 0.0
        # for p in model.parameters():
        #     print(p.size(),':', torch.sum(p))
        #     _sum += torch.sum(p)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        agent._step_updates(*batch, auxiliary_info=None, iteration=i)
        if (i) % 100 == 0:
          agent._eval(env, n_episodes=5)
    agent._eval(env, n_episodes=5)
    torch.save(agent.model_target.state_dict(), '/home/ml/svenka10/debug-data/params_dict')

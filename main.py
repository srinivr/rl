from agents.nstep_dqn_agent import NStepSynchronousDQNAgent
from models.classic_control.simple_cartpole_model import SimpleCartPoleModel
import gym
import envs.treeqn.push
from agents.dqn_agent import DQNAgent
from models.treeqn.tree_qn import PushModel
from utils.scheduler.LinearScheduler import LinearScheduler
from utils.scheduler.decay_scheduler import DecayScheduler
from utils.vec_env.subproc_vec_env import SubprocVecEnv


def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        # print('max_steps:', env._max_episode_steps)
        env.seed(seed)
        return env

    return _f


experiment = 'CartPoleNStepSynchronousDQN'
device = 'cpu'

if experiment == 'CartPoleDQN':
    env = gym.make('CartPole-v0')
    agent = DQNAgent(SimpleCartPoleModel, [4, 2], None, n_episodes=50000, replay_buffer_size=100000, device=device,
                     epsilon_scheduler_use_steps=True, target_synchronize_steps=10000, grad_clamp=[-1, 1],
                     training_evaluation_frequency=100)
    agent.learn(env)

elif experiment == 'CartPoleNStepSynchronousDQN':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    nproc = 8
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    agent = NStepSynchronousDQNAgent(SimpleCartPoleModel, [4, 2], None, n_envs=nproc, device=device,
                                     target_synchronize_steps=10000, grad_clamp=[-1, 1], training_evaluation_frequency=10000)
    agent.learn(envs, env)

elif experiment == 'PushNStepSyncDQN':
    env_name = 'Push-v0'
    env = gym.make(env_name)
    nproc = 8
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    # params
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = NStepSynchronousDQNAgent(PushModel, [5, 4, 2], None, n_envs=nproc, device=device,
                                     optimizer_parameters=optimizer_parameters, target_synchronize_steps=40000,
                                     grad_clamp=[-1, 1], training_evaluation_frequency=40000,
                                     epsilon_scheduler=LinearScheduler())
    agent.learn(envs, env)

elif experiment == 'PushDQN':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    optimizer_parameters = {'lr': 1e-4, 'alpha': 0.99, 'eps': 1e-5}
    agent = DQNAgent(PushModel, [5, 4, 2], None, device=device, optimizer_parameters=optimizer_parameters,
                     target_synchronize_steps=40000, grad_clamp=[-1, 1], training_evaluation_frequency=40000,
                     epsilon_scheduler=LinearScheduler(), epsilon_scheduler_use_steps=True
                     )
    agent.learn(env, env)

else:
    raise IndexError

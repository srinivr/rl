from agents.nstep_dqn_agent import NStepSynchronousDQNAgent
from models.classic_control.simple_cartpole_model import SimpleCartPoleModel
import gym
import envs.treeqn.push
from agents.dqn_agent import DQNAgent
from models.treeqn.tree_qn import PushModel
from utils.scheduler.decay_scheduler import StepDecayScheduler
from utils.vec_env.subproc_vec_env import SubprocVecEnv





def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        # print('max_steps:', env._max_episode_steps)
        env.seed(seed)
        return env

    return _f

if False:
    env = gym.make('CartPole-v0')
    agent = DQNAgent(SimpleCartPoleModel, [4, 2], None, n_episodes=50000, replay_buffer_size=100000, epsilon_scheduler_use_steps=True,
                     target_update_steps=10000, grad_clamp=[-1, 1], evaluation_frequency=100)
    agent.learn(env)
    #agent._eval(env, 0.5)
elif False:
    #env = gym.make('CartPole-v0')

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    nproc  = 8
    envs = [make_env(env_name, 2 * seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    agent = NStepSynchronousDQNAgent(SimpleCartPoleModel, [4, 2], None, n_envs=nproc, device='cpu',
                                     target_update_steps=10000, grad_clamp=[-1, 1], evaluation_frequency=10000)
    agent.learn(envs, env)

else:
    env_name = 'Push-v0'
    env = gym.make(env_name)
    nproc = 8
    envs = [make_env(env_name, seed) for seed in range(nproc)]

    envs = SubprocVecEnv(envs)
    agent = NStepSynchronousDQNAgent(PushModel, [5, 4, 2], None, n_envs=nproc, device='cpu',
                                     target_update_steps=10000, grad_clamp=[-1, 1], evaluation_frequency=10000)
    agent.learn(envs, env)

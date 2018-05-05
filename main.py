from models.classic_control.simple_cartpole_model import SimpleCartPoleModel
import gym
from agents.dqn_agent import DQNAgent

env = gym.make('CartPole-v0')
agent = DQNAgent(SimpleCartPoleModel, [4, 2], None, replay_buffer_size=10000, epsilon_scheduler_use_steps=False,
                 target_update_frequency=200, grad_clamp=[-1, 1])
agent.learn(env)

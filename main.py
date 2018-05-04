from models.classic_control.simple_cartpole_agent import SimpleCartPoleAgent
import gym
from agents.dqn_agent import DQNAgent

env = gym.make('CartPole-v0')
agent = DQNAgent(SimpleCartPoleAgent, [2, 4], env, None, )
print('Agent created successfully')

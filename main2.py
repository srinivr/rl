import numpy as np
from numpy import random

random.seed(1234)
import gym

w = np.random.rand(2, 4)
env = gym.make('CartPole-v0')
eps = .5
lr = 0.0001
returns = []
for ep in range(1000):
    o = env.reset()
    done = False
    ret = 0.
    while not done:
        # env.render()
        if ep >= 1800:
            env.render()
        if random.random() < eps:
            action = random.randint(0, 2)
        else:
            action = np.argmax(np.dot(w, o))
        _o, rew, done, info = env.step(action)

        ret += rew
        target = rew +  np.max(np.dot(w, _o))
        if done:
            target = rew
        td_error = target - np.dot(w, o)[action]
        w[action] = w[action] + lr * (td_error * o)
        o = _o
        eps *= 0.995
    returns.append(ret)
    print('episode', ep, 'avg return:', np.mean(returns[-100:]))

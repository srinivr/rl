from gym.envs.registration import register

register(
    id='Push-v0',
    entry_point='envs.treeqn.push:Push',
)
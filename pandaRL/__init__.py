from gym.envs.registration import register

register(
    id='pandaReach-v0',
    entry_point='pandaRL.envs:pandaReach',
)


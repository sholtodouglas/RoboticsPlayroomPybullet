from gym.envs.registration import register

register(
    id='pandaReach-v0',
    entry_point='pandaRL.envs:pandaReach',
)

register(
    id='pointMass3D-v0',
    entry_point='pandaRL.envs:pointMassEnv',
)

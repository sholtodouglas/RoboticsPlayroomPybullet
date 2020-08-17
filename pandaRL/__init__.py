from gym.envs.registration import register

register(
    id='pandaReach-v0',
    entry_point='pandaRL.envs:pandaReach',
)

register(
    id='pandaReach2D-v0',
    entry_point='pandaRL.envs:pandaReach2D',
)

register(
    id='pointMass3D-v0',
    entry_point='pandaRL.envs:pointMassEnv',
)

register(
    id='pandaPush-v0',
    entry_point='pandaRL.envs:pandaPush',
)

register(
    id='pandaPick-v0',
    entry_point='pandaRL.envs:pandaPick',
)

register(
    id='pandaPlay-v0',
    entry_point='pandaRL.envs:pandaPlay',
)

register(
    id='pandaPlayJoints-v0',
    entry_point='pandaRL.envs:pandaPlayRelJoints',
)

register(
    id='pandaPlayJoints1Obj-v0',
    entry_point='pandaRL.envs:pandaPlayRelJoints1Obj',
)

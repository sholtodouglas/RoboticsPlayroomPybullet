from gym.envs.registration import register

register(
    id='pandaReach-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaReach',
)

register(
    id='pandaReach2D-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaReach2D',
)

register(
    id='pointMass3D-v0',
    entry_point='roboticsPlayroomPybullet.envs:pointMassEnv',
)

register(
    id='pandaPush-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPush',
)

register(
    id='pandaPick-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPick',
)

register(
    id='pandaPlay-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPlay',
)
register(
    id='pandaPlay1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPlay1Obj',
)
register(
    id='pandaPlayRel1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPlayRel1Obj',
)
register(
    id='pandaPlayJoints-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPlayRelJoints',
)

register(
    id='pandaPlayRelJoints1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPlayRelJoints1Obj',
)

register(
    id='pandaPlayAbsJoints1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPlayAbsJoints1Obj',
)

register(
    id='pandaPlayAbsRPY1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPlayAbsRPY1Obj',
)

register(
    id='pandaPlayRelRPY1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:pandaPlayRelRPY1Obj',
)


register(
    id='UR5Play1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:UR5Play1Obj',
)

register(
    id='UR5PlayRel1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:UR5PlayRel1Obj',
)

register(
    id='UR5PlayRelJoints1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:UR5PlayRelJoints1Obj',
)

register(
    id='UR5PlayAbsJoints1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:UR5PlayAbsJoints1Obj',
)

register(
    id='UR5PlayAbsRPY1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:UR5PlayAbsRPY1Obj',
)

register(
    id='UR5PlayRelRPY1Obj-v0',
    entry_point='roboticsPlayroomPybullet.envs:UR5PlayRelRPY1Obj',
)
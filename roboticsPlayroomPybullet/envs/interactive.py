
from envList import * 
import numpy as np
import pybullet as p

def add_xyz_rpy_controls(env):
    controls = []
    orn = env.instance.default_arm_orn_RPY
    controls.append(env.p.addUserDebugParameter("X", -1, 1, 0))
    controls.append(env.p.addUserDebugParameter("Y", -1, 1, 0.00))
    controls.append(env.p.addUserDebugParameter("Z", -1, 1, 0.2))
    controls.append(env.p.addUserDebugParameter("R", -4, 4, orn[0]))
    controls.append(env.p.addUserDebugParameter("P", -4, 4, orn[1]))
    controls.append(env.p.addUserDebugParameter("Y", -4,4, orn[2]))
    controls.append(env.p.addUserDebugParameter("grip", env.action_space.low[-1], env.action_space.high[-1], 0))
    return controls

def add_joint_controls(env):
    for i, obj in enumerate(env.instance.restJointPositions):
        env.p.addUserDebugParameter(str(i), -2*np.pi, 2*np.pi, obj)


joint_control = False #Toggle this flag to control joints or ABS RPY Space
def main():
    
    env_type = {False: UR5PlayAbsRPY1Obj , True: UR5PlayAbsJoints1Obj}
    env = env_type[joint_control]()
    
    env.render(mode='human')
    env.reset()
    if joint_control:
        add_joint_controls(env)
    else:
        controls = add_xyz_rpy_controls(env)

    env.render(mode='human')
    env.reset()

    for i in range(1000000):

        if joint_control:
            poses  = []
            for i in range(0, len(env.instance.restJointPositions)):
                poses.append(env.p.readUserDebugParameter(i))
            # Uses a hard reset of the arm joints so that we can quickly debug without worrying about forces
            env.instance.reset_arm_joints(env.instance.arm, poses)

        else:
            action = []
            for i in range(0, len(controls)):
                action.append(env.p.readUserDebugParameter(i))

            state = env.instance.calc_actor_state()
            obs, r, done, info = env.step(np.array(action))
             
            # Shows the block position just above where it is
            # x = obs['achieved_goal']
            # x[2] += 0.1
            # env.visualise_sub_goal(obs['achieved_goal'], sub_goal_state='achieved_goal')

if __name__ == "__main__":
    main()

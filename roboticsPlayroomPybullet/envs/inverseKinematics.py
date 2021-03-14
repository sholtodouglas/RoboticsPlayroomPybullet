
import pybullet as p
import numpy as np
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
from pybullet_utils import bullet_client
os.sys.path.insert(0, currentdir)

class InverseKinematicsSolver():
    '''
    A class to do multiple steps of IK using an arm at baseline, which means it converges onto one solution and is less finnicky with rpy
    While this isn't necessary for the panda, for some reason it is with the UR5
    '''

    def __init__(self, base_pos, base_orn, ee_index, default_joints):
        self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.ur5 = self.p.loadURDF(currentdir + "/ur_e_description/ur5e2.urdf",
                                                 base_pos,
                                                 base_orn, useFixedBase=True)
        self.n_joints = self.p.getNumJoints(self.ur5)

        # [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint,wrist_3_joint]
        self.joints_indices = [0,1,2,3,4,5]
        self.jointNames = [self.p.getJointInfo(self.ur5, i)[1] for i in self.joints_indices]

        self.arm_joint_positions = None
        self.arm_joint_velocity = None

        self.default_joints = default_joints
        self.ee_index=ee_index
        self.set_states(self.default_joints)

    # sets the arm to refernce jointPositions
    def set_states(self,states):
        for idx, i in enumerate(self.joints_indices):
            # print_output("states {}".format(i))
            self.p.resetJointState(self.ur5, i, states[idx])

    # Gets position of solving UR5 - used for debugging
    def get_position(self):
        return self.p.getLinkState(self.ur5,self.ee_index)[0:2]

    def calc_angles(self, pos,ori, current_states):
        #always set back to the ideal original state so we don't end up with funny angles.
        self.set_states(current_states)
        for i in range(0,3): # converge on the solution
            angles = self.p.calculateInverseKinematics(self.ur5,self.ee_index, pos,ori)[0:6]
            self.set_states(angles)
        return self.p.calculateInverseKinematics(self.ur5,self.ee_index, pos,ori)[0:6]


import numpy as np


def goto_joint_poses(self, jointPoses, gripper):
    indexes = [i for i in range(self.numDofs)]
    index_len = len(indexes)
    local_ll, local_ul = None, None
    if self.arm_type == 'Panda':
        local_ll = np.array([-0.6, -2.2, -3.0, -3.04878596, -np.pi,-np.pi, -np.pi, -np.pi])
        local_ul = np.array([ 3,  1.8, 0.5, -0.5002492,  3.,3.45266257,  2.40072908,  np.pi        ])
        inc = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2])
    elif self.arm_type == 'UR5':
        local_ll = np.array([-np.pi*2]*self.numDofs)
        local_ul = np.array([-0.7,np.pi*2,-0.5,np.pi*2,np.pi*2,np.pi*2])
        inc = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    targetPoses = np.clip(np.array(jointPoses[:index_len]), local_ll[:index_len], local_ul[:index_len])


    current_poses = np.array([self.bullet_client.getJointState(self.arm, j)[0] for j in range(index_len)])
    targetPoses = np.clip(targetPoses, current_poses-inc, current_poses+inc)
    self.bullet_client.setJointMotorControlArray(self.arm, indexes, self.bullet_client.POSITION_CONTROL,
                                                    targetPositions=targetPoses ,
                                                    forces=[240.]*len(indexes))

    if gripper is not None:
        self.close_gripper(gripper)

    return targetPoses


def goto(self, pos=None, orn=None, gripper=None):

    if pos is not None and orn is not None:


        pos = self.add_centering_offset(pos)
        if self.arm_type == 'Panda':
            jointPoses = self.bullet_client.calculateInverseKinematics(self.arm, self.endEffectorIndex, pos, orn, self.ll ,
                                                                        self.ul,
                                                                        self.jr, self.restJointPositions, maxNumIterations=200)
        elif self.arm_type == 'UR5':
            current_poses = np.array([self.bullet_client.getJointState(self.arm, j)[0] for j in range(self.numDofs)])
            # print(current_poses)
            jointPoses = self.IKSolver.calc_angles(pos,orn, current_poses)
            # jointPoses = self.bullet_client.calculateInverseKinematics(self.arm, self.endEffectorIndex, pos, orn, ll,
            #                                                        ul,
            #                                                        jr, self.restJointPositions, maxNumIterations=200)

        targetPoses = self.goto_joint_poses(jointPoses, gripper)
    return targetPoses

def absolute_quat_step(self, action):

    assert len(action) == 8
    
    new_pos = action[0:3]
    gripper = action[-1]
    targetPoses = self.goto(new_pos, action[3:7], gripper)
    return targetPoses

    # this function is only for fully funcitonal robots, will have ori and gripper
def relative_quat_step(self, action):
    assert len(action) == 8

    current_pos = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[0]
    current_orn = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[1]
    new_pos = action[0:3] + current_pos
    new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
    new_orn = action[3:7] + current_orn
    gripper = action[-1]
    targetPoses = self.goto(new_pos, new_orn, gripper)
    return targetPoses

def absolute_rpy_step(self, action):
    assert len(action) == 7
    new_pos = action[0:3]
    #new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
    new_orn = action[3:6]
    gripper = action[-1]
    targetPoses = self.goto(new_pos, self.bullet_client.getQuaternionFromEuler(new_orn), gripper)
    return targetPoses

def relative_rpy_step(self, action):
    assert len(action) == 7

    current_pos = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[0]
    current_orn = self.bullet_client.getEulerFromQuaternion(self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[1])
    new_pos = action[0:3] + current_pos
    new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
    new_orn = action[3:6] + current_orn
    gripper = action[-1]
    targetPoses = self.goto(new_pos, self.bullet_client.getQuaternionFromEuler(new_orn), gripper)
    return targetPoses

    # take a step with an action commanded in joint space # this doesn't yet have first class support judt trying hey
def relative_joint_step(self, action):
    current_poses = np.array([self.bullet_client.getJointState(self.arm, j)[0] for j in range(self.numDofs)])
    jointPoses = action[:-1] + current_poses
    gripper = action[-1]
    targetPoses = self.goto_joint_poses(jointPoses, gripper)
    return targetPoses

def absolute_joint_step(self, action):
    targetPoses = self.goto_joint_poses( action[:-1], action[-1])
    return targetPoses


# 0 is most open, 1 is least
def close_gripper(self, amount):
    '''
    0 : open grippeer
    1 : closed gripper
    '''
    if self.arm_type == 'Panda':
        amount = 0.04 - amount/25 # magic numbers, magic numbers everywhere!
        for i in [9, 10]:

            self.bullet_client.setJointMotorControl2(self.arm, i, self.bullet_client.POSITION_CONTROL, amount,
                                                        force=100)
    else:
    # left/ right driver appears to close at 0.03
        amount -= 0.2
        driver = amount * 0.055

        self.bullet_client.setJointMotorControl2(self.arm, 18, self.bullet_client.POSITION_CONTROL, driver,
                                                    force=100)
        left = self.bullet_client.getJointState(self.arm, 18)[0]
        self.bullet_client.setJointMotorControl2(self.arm, 20, self.bullet_client.POSITION_CONTROL, left,
                                                    force=1000)


        #self.bullet_client.resetJointState(self.arm, 20, left)

        spring_link = amount * 0.5
        self.bullet_client.setJointMotorControl2(self.arm, 12, self.bullet_client.POSITION_CONTROL, spring_link,
                                                    force=100)
        self.bullet_client.setJointMotorControl2(self.arm, 15, self.bullet_client.POSITION_CONTROL, spring_link,
                                                    force=100)


        driver_mimic = amount * 0.8
        self.bullet_client.setJointMotorControl2(self.arm, 10, self.bullet_client.POSITION_CONTROL, driver_mimic,
                                                force=100)
        self.bullet_client.setJointMotorControl2(self.arm, 13, self.bullet_client.POSITION_CONTROL, driver_mimic,
                                                    force=100)

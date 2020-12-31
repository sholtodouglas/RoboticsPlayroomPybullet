
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import gym, gym.utils, gym.utils.seeding
import pybullet as p
import numpy as np
import pybullet_data
import time
from pybullet_utils import bullet_client
import pybullet_data as pd
urdfRoot = pybullet_data.getDataPath()
import gym.spaces as spaces
import math
from scenes import *
from shadow_arm import InverseKinematicsSolver
GUI = False
viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0.25, 0], distance=1.0, yaw=130, pitch=-45, roll=0,
                                                 upAxisIndex=2)

projectionMatrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=10)

useNullSpace = 1
ikSolver = 1



# restposes for null space

arm_z_min = -0.055




def gripper_camera(bullet_client, pos, ori):
    # Center of mass position and orientation (of link-7)
    pos = np.array(pos)
    ori = p.getEulerFromQuaternion(ori) + np.array([0,-np.pi/2,0])
    ori = p.getQuaternionFromEuler(ori)
    rot_matrix = bullet_client.getMatrixFromQuaternion(ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    camera_target = (1, 0, 0) # z-axis
    init_up_vector = (0, 0, 1) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(camera_target)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix_gripper = bullet_client.computeViewMatrix(pos, pos + camera_vector, up_vector)
    img = bullet_client.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix,shadow=0, flags = bullet_client.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return img

class pointMassSim():

    def __init__(self, bullet_client, offset, load_scene, arm_lower_lim, arm_upper_lim,
        env_lower_bound, env_upper_bound, goal_lower_bound,
        goal_upper_bound, obj_lower_bound, obj_upper_bound, use_orientation, return_velocity, render_scene, pointMass = False, 
        fixed_gripper = False, play=False, show_goal=True, num_objects =0, arm_type = 'Panda' ):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        
        if play:
            self.objects, self.drawer, self.joints, self.toggles = load_scene(self.bullet_client, offset, flags, env_lower_bound, env_upper_bound, num_objects) # Todo: later, put this after the centering offset so that objects are centered around it too.
        else:
            self.objects = load_scene(self.bullet_client, offset, flags, env_lower_bound, env_upper_bound)
        self.num_objects = num_objects
        self.use_orientation = use_orientation
        self.return_velocity = return_velocity
        self.num_objects = len(self.objects)
        self.num_goals = max(self.num_objects, 1)
        self.pointMass = pointMass
        self.arm_lower_lim = arm_lower_lim
        self.arm_upper_lim = arm_upper_lim
        self.env_upper_bound = env_upper_bound
        self.env_lower_bound = env_lower_bound
        self.goal_upper_bound = goal_upper_bound
        self.goal_lower_bound = goal_lower_bound
        self.obj_lower_bound = obj_lower_bound
        self.obj_upper_bound = obj_upper_bound
        self.render_scene = render_scene
        self.play = play
        self.physics_client_active = 0
        self.movable_goal = False
        self.roving_goal = False
        self.sub_goals = None
        self.fixed_gripper = fixed_gripper
        self.arm_type = arm_type
        if self.arm_type == 'Panda':
            self.default_arm_orn_RPY = [0, 0,0]
            self.default_arm_orn = self.bullet_client.getQuaternionFromEuler(self.default_arm_orn_RPY)
            self.init_arm_base_orn = p.getQuaternionFromEuler([0,0,math.pi/2])
            self.endEffectorIndex = 11
            self.restJointPositions = [-0.529, 0.463, 0.265, -2.183, 3.042, 1.984, 0.728, 0.02, 0.02]
            self.numDofs = 7
        elif self.arm_type == 'UR5':
            self.default_arm_orn_RPY = [0, 0, 0]
            self.default_arm_orn = self.bullet_client.getQuaternionFromEuler(self.default_arm_orn_RPY)
            self.init_arm_base_orn = p.getQuaternionFromEuler([0,0,math.pi/2])
            self.endEffectorIndex = 7
            # this pose makes it very easy for the IK to do the 'underhand' grip, which isn't well solved for
            # if we take an over hand top down as our default (its very easy for it to flip into an unideal configuration)

            self.restJointPositions = [-1.587, -2.116, -1.587, -2.5, -0.066, 0, 0]
            self.numDofs = 6


                
        else:
            raise NotImplementedError
        self.ll = [-7] * self.numDofs
        # upper limits for null space (todo: set them to proper range)
        self.ul = [7] * self.numDofs
        # joint ranges for null space (todo: set them to proper range)
        self.jr = [6] * self.numDofs
        self.record_images = False
        self.last_obs = None  # use for quaternion flipping purpposes (with equivalent quaternions)
        self.last_ag = None

        sphereRadius = 0.03
        mass = 1
        colSphereId = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_SPHERE, radius=sphereRadius)
        visId = self.bullet_client.createVisualShape(self.bullet_client.GEOM_SPHERE, radius=sphereRadius,
                                                     rgbaColor=[1, 0, 0, 1])
        centering_offset =np.array([0, 0.0, 0.0])
        self.original_offset = offset  # exclusively for if we span more arms for visulaising sub goals
        if self.pointMass:

            self.offset = offset +  centering_offset # to center the env about the panda gripper location
            init_loc = self.add_centering_offset(np.array([0, 0.0, 0.1]))
            self.mass = self.bullet_client.createMultiBody(mass, colSphereId, visId, init_loc)
            # collisionFilterGroup = 0
            # collisionFilterMask = 0
            # self.bullet_client.setCollisionFilterGroupMask(self.mass, -1, collisionFilterGroup,
            #                                                collisionFilterMask)


            self.mass_cid = self.bullet_client.createConstraint(self.mass, -1, -1, -1, self.bullet_client.JOINT_FIXED,
                                                                [0, 0, 0], [0, 0, 0],
                                                                init_loc, [0, 0, 0, 1])

        else:
            print(currentdir)

            global ll
            global ul
            if self.play:

                if self.arm_type == 'Panda':
                    self.restJointPositions = [-0.6, 0.437, 0.217, -2.09, 1.1, 1.4, 1.3, 0.0, 0.0, 0.0]
                    #self.restJointPositions = [0.03, 0.92, -0.02, -2.36, 1.68, 1.28, 0.76, 0.00, 0.0, 0.0]
                    self.init_arm_base_pos =  np.array([-0.5, 0.0, -0.05])
                    self.init_arm_base_orn = p.getQuaternionFromEuler([0, 0, 0])
                elif self.arm_type == 'UR5':
                    self.restJointPositions =  [-1.50189075, - 1.6291067, - 1.87020409, - 1.21324173, 1.57003561, 0.06970189]
                    self.init_arm_base_pos = np.array([0.5, -0.1, 0.0])
                    self.init_arm_base_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])

            else:
 
                self.init_arm_base_pos = np.array([0, 0, 0])

            if self.arm_type == 'Panda':

                self.panda = self.bullet_client.loadURDF(currentdir + "/franka_panda/panda.urdf",
                                                         self.init_arm_base_pos + offset,
                                                         self.init_arm_base_orn , useFixedBase=True, flags=flags)
                c = self.bullet_client.createConstraint(self.panda,
                                                        9,
                                                        self.panda,
                                                        10,
                                                        jointType=self.bullet_client.JOINT_GEAR,
                                                        jointAxis=[1, 0, 0],
                                                        parentFramePosition=[0, 0, 0],
                                                        childFramePosition=[0, 0, 0])
                self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

            elif self.arm_type == 'UR5':

                self.panda = self.bullet_client.loadURDF(currentdir + "/ur_e_description/ur5e2.urdf",
                                                         self.init_arm_base_pos + offset,
                                                         self.init_arm_base_orn, useFixedBase=True, flags=flags)
                self.IKSolver  = InverseKinematicsSolver( self.init_arm_base_pos, self.init_arm_base_orn, self.endEffectorIndex, self.restJointPositions)
                # c = self.bullet_client.createConstraint(self.panda,
                #                                         18,
                #                                         self.panda,
                #                                         20,
                #                                         jointType=self.bullet_client.JOINT_GEAR,
                #                                         jointAxis=[1, 0, 0],
                #                                         parentFramePosition=[0, 0, 0],
                #                                         childFramePosition=[0, 0, 0])
                # self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)


            else:
                raise NotImplementedError

            self.offset = offset + centering_offset  # to center the env about the panda gripper location
            # create a constraint to keep the fingers centered

            for j in range(self.bullet_client.getNumJoints(self.panda)):
                self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)


        self.state = 0
        self.control_dt = 1. / 240.
        self.finger_target = 0
        self.gripper_height = 0.2
        # create the goal objects
        self.show_goal = show_goal
        alpha = 1
        self.obj_colors = [[0, 1, 0, alpha], [0, 0, 1, alpha]]  # colors for two objects
        if self.render_scene and self.show_goal:


            relativeChildOrientation = [0, 0, 0, 1]
            self.goals = []
            self.goal_cids = []
            collisionFilterGroup = 0
            collisionFilterMask = 0
            for g in range(self.num_goals):
                init_loc = self.add_centering_offset(np.array([0, 0.0, 0.2]))
                visId = self.bullet_client.createVisualShape(self.bullet_client.GEOM_SPHERE, radius=sphereRadius,
                                                             rgbaColor=self.obj_colors[g])
                self.goals.append(self.bullet_client.createMultiBody(mass, colSphereId, visId, init_loc))

                self.bullet_client.setCollisionFilterGroupMask(self.goals[g], -1, collisionFilterGroup,
                                                               collisionFilterMask)
                self.goal_cids.append(
                    self.bullet_client.createConstraint(self.goals[g], -1, -1, -1, self.bullet_client.JOINT_FIXED,
                                                        [0, 0, 0], [0, 0, 0],
                                                        init_loc, relativeChildOrientation))
        for i in range(self.num_objects):
            self.bullet_client.changeVisualShape(self.objects[i], -1, rgbaColor=self.obj_colors[i])




    def add_centering_offset(self, numbers):
        # numbers either come in as xyz, or xyzxyzxyz etc (in the case of goals or achieved goals)
        offset = np.array(list(self.offset) * (len(numbers) // 3))
        numbers = numbers + offset
        return numbers

    def subtract_centering_offset(self, numbers):
        offset = np.array(list(self.offset) * (len(numbers) // 3))
        numbers = numbers - offset
        return numbers

    def reset_goal_pos(self, goal=None):
        if goal is None:
            self.goal = []
            for g in range(self.num_goals):
                goal = np.random.uniform(self.goal_lower_bound, self.goal_upper_bound)
                self.goal.append(goal)
            self.goal = np.concatenate(self.goal)
        else:

            self.goal = np.array(goal)
        if self.render_scene and self.show_goal:
            index = 0

            for g in range(self.num_goals):
                pos = self.add_centering_offset(self.goal[index:index + 3])
                self.bullet_client.resetBasePositionAndOrientation(self.goals[g], pos, [0, 0, 0, 1])
                self.bullet_client.changeConstraint(self.goal_cids[g], pos, maxForce=100)
                index += 3


    def reset_object_pos(self, obs=None):
        # Todo object velocities to make this properly deterministic
        if self.play:
            self.bullet_client.resetBasePositionAndOrientation(self.drawer['drawer'], self.drawer['defaults']['pos'],
                                                               self.drawer['defaults']['ori'])
            for i in self.joints:
                self.bullet_client.resetJointState(i, 0, 0) # reset drawer, button etc

        if obs is None:
            height_offset = 0.03
            for o in self.objects:
                pos = self.add_centering_offset(np.random.uniform(self.obj_lower_bound, self.obj_upper_bound))
                pos[2] = pos[2] + height_offset  # so they don't collide
                self.bullet_client.resetBasePositionAndOrientation(o, pos, [0,0,0,1])
                height_offset += 0.03
            for i in range(0, 100):
                self.bullet_client.stepSimulation() # let everything fall into place, falling in to piecees...
            for o in self.objects:
                #print(self.env_upper_bound, self.bullet_client.getBasePositionAndOrientation(o)[0])
                if (self.subtract_centering_offset(self.bullet_client.getBasePositionAndOrientation(o)[0]) > self.env_upper_bound).any():
                    self.reset_object_pos()

        else:

            if self.use_orientation:
                index = 11
                increment = 10
            else:
                index = 7
                increment = 6
            for o in self.objects:
                pos = obs[index:index + 3]
                if self.use_orientation:
                    orn = obs[index + 3:index + 7]
                else:
                    orn =  [0,0,0,1]
                self.bullet_client.resetBasePositionAndOrientation(o, self.add_centering_offset(pos), orn)
                index += increment


    def updateToggles(self):
        for k, v in self.toggles.items():
            jointstate = self.bullet_client.getJointState(k, 0)[0]
            if v[0] == 'button':

                if jointstate < 0.025:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1,0,0,1])
                else:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])
            if v[0] == 'dial':

                if dial_to_0_1_range(jointstate) < 0.5:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1,0,0,1])
                else:
                    self.bullet_client.changeVisualShape(v[1], -1, rgbaColor=[1, 1, 1, 1])



    def runSimulation(self):
        # also do toggle updating here
        self.updateToggles() # so its got both in VR and replay out
        #for i in range(0, 18): # 20Hz control with 480 timestep
        for i in range(0, 12):  # 25Hz control at 300
            self.bullet_client.stepSimulation()



    def reset_arm_joints(self, arm, poses):
        index = 0

        for j in range(len(poses)):
            self.bullet_client.changeDynamics(arm, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(arm, j)
            # print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(arm, j, poses[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(arm, j, poses[index])
                index = index + 1

    def reset_pointMass(self, o = None):
        if  o is None:
            new_pos = self.add_centering_offset(np.random.uniform(self.goal_lower_bound,self.goal_upper_bound))
        else:
            new_pos = self.add_centering_offset(o[0:3])
        self.bullet_client.resetBasePositionAndOrientation(self.mass,
                                                           new_pos, [0, 0, 0, 1])
        self.bullet_client.changeConstraint(self.mass_cid, new_pos, maxForce=100)




    def reset_arm(self, which_arm = None, o = None, from_init=True):

        orn = self.default_arm_orn
        if  o is None:
            new_pos = self.add_centering_offset(np.random.uniform(self.goal_lower_bound,self.goal_upper_bound))
            if  self.arm_type == 'UR5':
                new_pos[2] = new_pos[2] +0.2
        else:
            new_pos = self.add_centering_offset(o[0:3])
            if self.use_orientation:
                if self.return_velocity:
                    orn = o[6:10] # because both pos and pos_vel are in the state
                else:
                    orn = o[3:7]

        if from_init:

            self.reset_arm_joints(which_arm, self.restJointPositions) # put it into a good init for IK
        #orn = p.getQuaternionFromEuler([0, 0, 0])
        #new_pos = (0, 0.4, 0.3)
        #jointPoses = self.bullet_client.calculateInverseKinematics(arm, self.endEffectorIndex, new_pos, orn, ll, ul, jr, rp, maxNumIterations=200)
        jointPoses = self.bullet_client.calculateInverseKinematics(which_arm, self.endEffectorIndex, new_pos, orn,)[0:6]
        #jointPoses = self.bullet_client.calculateInverseKinematics(arm, self.endEffectorIndex, new_pos, orn, maxNumIterations=100)

        self.reset_arm_joints(which_arm, jointPoses)



    def reset(self, o = None):
        if o is not None:
            self.reset_object_pos(o)
            if self.pointMass:
                self.reset_pointMass(o)
            else:
                self.reset_arm(self.panda, o)

            self.reset_goal_pos()  # o[-self.num_goals*3:])
        else:
            self.reset_object_pos()
            if self.pointMass:
                self.reset_pointMass()
            else:
                self.reset_arm(self.panda)

            self.reset_goal_pos()
        self.t = 0

    def visualise_sub_goal(self, sub_goal, sub_goal_state = 'full_positional_state'): # Todo this does not yet account for if we would like to do subgoals with orientation.

        if self.sub_goals is None:
            # in the case of ag, num objects = 0 we want just ghost arm
            # ag, num object > 1, we want spheres per object
            # in the case of controllable we just want ghost arm pos
            # in the case of full positional, we want ghost arm + num objects sphere
            self.sub_goals = []
            collisionFilterGroup = 0
            collisionFilterMask = 0
            if sub_goal_state == 'full_positional_state' or sub_goal_state == 'controllable_achieved_goal':
                flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                #flags = self.bullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                if self.arm_type == 'Panda':
                    self.ghost_arm = self.bullet_client.loadURDF(
                        os.path.dirname(os.path.abspath(__file__)) + "/franka_panda/ghost_panda.urdf", self.init_arm_base_pos + self.original_offset,
                        self.init_arm_base_orn,  useFixedBase = True, flags=flags)
                else:
                    raise NotImplementedError

                self.bullet_client.setCollisionFilterGroupMask(self.ghost_panda, -1, collisionFilterGroup,collisionFilterMask)
                for i in range(0, self.bullet_client.getNumJoints(self.ghost_arm)):
                    self.bullet_client.setCollisionFilterGroupMask(self.ghost_arm, i, collisionFilterGroup,
                                                                   collisionFilterMask)
                self.reset_arm_joints(self.ghost_arm, self.restJointPositions)  # put it into a good init for IK

            if sub_goal_state == 'full_positional_state' or sub_goal_state is 'achieved_goal':
                sphereRadius = 0.03
                mass = 0
                colSphereId = self.bullet_client.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
                if sub_goal_state == 'full_positional_state':
                    if self.use_orientation:
                        index = 8
                    else:
                        index = 4
                if sub_goal_state == 'achieved_goal':
                    index = 0
                for i in range(0, self.num_objects):
                    color = self.obj_colors[i]
                    color[3] = 0.5 # set alpha to 0.5 for ghostly subgoal appearance
                    if self.play:
                        extents = [0.025*2, 0.025, 0.025]
                    else:
                        extents = [0.03, 0.03, 0.03]

                    visId = p.createVisualShape(p.GEOM_BOX, halfExtents=extents,
                                                rgbaColor=color)
                    self.sub_goals.append(self.bullet_client.createMultiBody(mass, colSphereId, visId, sub_goal[index:index+3]))
                    self.bullet_client.setCollisionFilterGroupMask(self.sub_goals[i], -1, collisionFilterGroup,
                                                        collisionFilterMask)
                    index += 3

            if self.play:
                self.ghost_drawer = add_drawer(self.bullet_client, ghostly=True)
                door = add_door(self.bullet_client, ghostly=True)
                button, toggleSphere = add_button(self.bullet_client, ghostly=True)
                dial, toggleGrill = add_dial(self.bullet_client, ghostly=True)  # , thickness = thickness) 1.5

                self.ghost_joints = [door, button, dial]




        if sub_goal_state == 'controllable_achieved_goal':
            self.reset_arm(self.ghost_arm, sub_goal, from_init=False)
        elif sub_goal_state == 'full_positional_state':
            self.reset_arm(self.ghost_arm, sub_goal, from_init=False)
            if self.use_orientation:
                index = 8
            else:
                index = 4
        elif sub_goal_state == 'achieved_goal':
            index = 0

        if sub_goal_state != 'controllable_achieved_goal':
            for i in range(0, self.num_objects):
                if self.use_orientation:
                    self.bullet_client.resetBasePositionAndOrientation(self.sub_goals[i], self.add_centering_offset(
                        sub_goal[index:index + 3]), sub_goal[index+3:index + 7])
                    index += 7
                else:
                    self.bullet_client.resetBasePositionAndOrientation(self.sub_goals[i], self.add_centering_offset(sub_goal[index:index+3]), [0,0,0,1])
                    index += 3

        if self.play:
            drawer_pos = self.ghost_drawer['defaults']['pos']
            drawer_pos[1] = sub_goal[index]

            self.bullet_client.resetBasePositionAndOrientation(self.ghost_drawer['drawer'], drawer_pos, self.ghost_drawer['defaults']['ori'])
            index += 1
            for i, j in enumerate(self.ghost_joints):
                #(index+i)
                self.bullet_client.resetJointState(j, 0, sub_goal[index+i])  # reset drawer, button etc

    def delete_sub_goal(self):
        for i in self.sub_goals:
            self.bullet_client.removeBody(i)
        self.sub_goals = None
        for i in self.ghost_joints:
            self.bullet_client.removeBody(i)
        self.bullet_client.removeBody(self.ghost_drawer['drawer'])

        try:
            self.bullet_client.removeBody(self.ghost_arm)
        except:
            pass


    def gripper_proprioception(self):
        gripper_one = np.array(self.bullet_client.getLinkState(self.panda, 18)[0])
        gripper_two = np.array(self.bullet_client.getLinkState(self.panda, 20)[0])
        vector = gripper_two-gripper_one
        gripper_one = gripper_one + 0.2*vector

        try:
            obj_id, link_index, hit_fraction, hit_position, hit_normal = self.bullet_client.rayTest(gripper_one, gripper_two)[0]
            print(link_index, hit_fraction)
            #self.bullet_client.addUserDebugLine(gripper_one, gripper_two, [1,0,0], 0.5, 1)
            
            if link_index == 20:
                return 0 # free hand! :)
            else:
                return 1 # something in the way, oooh, something in the way
        else:
            return -1 # this shouldn't ever happen because the ray will always hit the other side of the gripper


    def calc_actor_state(self):

        if self.pointMass:
            pos = self.bullet_client.getBasePositionAndOrientation(self.mass)[0]
            vel = self.bullet_client.getBaseVelocity(self.mass)[0]
            orn, orn_vel, gripper_state = None, None, np.array([0.04])
        else:
            state = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)
            pos, orn, vel, orn_vel = state[0], state[1], state[-2], state[-1]

            if self.arm_type == 'Panda':
                gripper_state = [self.bullet_client.getJointState(self.panda, 9)[0]]
            else:
                gripper_state = [self.bullet_client.getJointState(self.panda, 18)[0]*23] # put it on a 0-1 scale


            joint_poses = [self.bullet_client.getJointState(self.panda, j)[0] for j in range(8)]

        #img = gripper_camera(self.bullet_client, pos, orn)

        return {'pos': self.subtract_centering_offset(pos), 'orn': orn, 'pos_vel': vel, 'orn_vel': orn_vel,
                'gripper': gripper_state, 'joints':joint_poses, 'gripper_proprioception': self.gripper_proprioception()}


    def calc_environment_state(self):
        object_states = {}
        for i in range(0, self.num_objects):
            pos, orn = self.bullet_client.getBasePositionAndOrientation(self.objects[i])
            vel = self.bullet_client.getBaseVelocity(self.objects[i])[0]
            object_states[i] = {'pos': self.subtract_centering_offset(pos), 'orn': orn, 'vel':vel}

        # get things like hinges, doors, dials, buttons etc
        i += 1
        if self.play:
            drawer_pos = self.bullet_client.getBasePositionAndOrientation(self.drawer['drawer'])[0][1] # get the y pos
            object_states[i] = {'pos': [drawer_pos], 'orn':[]}
            i += 1
            for j in range(0, len(self.joints)):
                data = self.bullet_client.getJointState(self.joints[j], 0)[0]
                if j == 2:
                    # this is the dial
                    data = dial_to_0_1_range(data) # and put it just slightly below -1, 1
                object_states[i+j] = {'pos':[data], 'orn':[]}

        return object_states

    def calc_state(self):
        self.updateToggles() # good place to update the toggles
        arm_state = self.calc_actor_state()
        arm_elements = ['pos']
        if self.return_velocity:
            arm_elements.append('pos_vel')
        if self.use_orientation:
            arm_elements.append('orn')
        arm_elements.append('gripper')
        state = np.concatenate([np.array(arm_state[i]) for i in arm_elements])
        if self.num_objects > 0:

            env_state = self.calc_environment_state()
            obj_elements = ['pos']
            if self.use_orientation:
                obj_elements.append('orn')
            if self.return_velocity:
                obj_elements.append('vel')

            obj_states = []
            for i, obj in env_state.items():
                o_s = []
                for key in obj_elements:
                    o_s += list(obj[key])
                obj_states.append(np.array(o_s))
                #obj_states.append(np.concatenate([obj[key] for key in obj_elements]))
            # TDOO: not have to write spearate code for getting the env
            obj_states = np.concatenate(obj_states)

            if self.use_orientation:
                state = np.concatenate([state, obj_states])
                achieved_goal = np.concatenate([np.array(list(obj['pos'])+ list(obj['orn'])) for (i, obj) in env_state.items()])
                full_positional_state = np.concatenate([arm_state['pos'], arm_state['orn'], arm_state['gripper'], achieved_goal])
            else:
                state = np.concatenate([state, obj_states])
                achieved_goal = np.concatenate([obj['pos'] for (i, obj) in env_state.items()])
                full_positional_state = np.concatenate([arm_state['pos'], arm_state['gripper'], achieved_goal])
        else:
            achieved_goal = arm_state['pos']
            full_positional_state = np.concatenate([arm_state['pos'], arm_state['gripper']])

        if self.record_images:
            img_arr = self.bullet_client.getCameraImage(200, 200, viewMatrix, projectionMatrix, flags=self.bullet_client.ER_NO_SEGMENTATION_MASK, shadow=0,
                                       renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)[2][:,:,:3] #just the rgb
        else:
            img_arr = None

        if self.play:
            state, achieved_goal = self.quaternion_safe_the_obs(state, achieved_goal)
        return_dict = {
            'observation': state.copy().astype('float32'),
            'achieved_goal': achieved_goal.copy().astype('float32'),
            'desired_goal': self.goal.copy().astype('float32'),
            'controllable_achieved_goal': np.concatenate([arm_state['pos'].copy(), arm_state['gripper'].copy()]).astype('float32'),
            # just the x,y,z pos of the self, the controllable aspects
            'full_positional_state': full_positional_state.copy().astype('float32'),
            'joints': arm_state['joints'],
            'velocity': np.concatenate([arm_state['pos_vel'], arm_state['orn_vel']]),
            'img': img_arr,
            'obs_rpy': np.concatenate([state[0:3], p.getEulerFromQuaternion(state[3:7]), state[7:]]).copy(),
            'gripper_proprioception':  arm_state['gripper_proprioception']
        }


        return return_dict

    def quaternion_safe_the_obs(self, obs, ag):

        def flip_quats(vector, last, idxs):
            for pair in idxs:
                quat = vector[pair[0]:pair[1]]
                last_quat = last[pair[0]:pair[1]]
                #print(np.sign(quat) == -np.sign(last_quat))
                if (np.sign(quat) == -np.sign(last_quat)).all():  # i.e, it is an equivalent quaternion
                    vector[pair[0]:pair[1]] = - vector[pair[0]:pair[1]]
            return vector


        if self.last_obs is None:
            pass
        else:
            indices = [(3,7), (11,15)] # self, and object one xyz q1-4 grip xyz q1-4
            if self.num_objects == 2:
                indices.append((19,23))
            obs = flip_quats(obs, self.last_obs, indices)
            indices =  [(3,7)] # just the objeect
            if self.num_objects == 2:
                indices.append((10,14))
            ag = flip_quats(ag, self.last_ag, indices)


        self.last_obs = obs
        self.last_ag = ag
        return obs, ag

    # 0 is most open, 1 is least
    def close_gripper(self, amount):
        '''
        0 : open grippeer
        1 : closed gripper
        '''
        if self.arm_type == 'Panda':
            amount = 0.04 - amount/25 # magic numbers, magic numbers everywhere!
            for i in [9, 10]:

                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, amount,
                                                         force=100)
        else:
        # left/ right driver appears to close at 0.03
            amount -= 0.2
            driver = amount * 0.055

            self.bullet_client.setJointMotorControl2(self.panda, 18, self.bullet_client.POSITION_CONTROL, driver,
                                                     force=100)
            left = self.bullet_client.getJointState(self.panda, 18)[0]
            self.bullet_client.setJointMotorControl2(self.panda, 20, self.bullet_client.POSITION_CONTROL, left,
                                                     force=1000)


            #self.bullet_client.resetJointState(self.panda, 20, left)

            spring_link = amount * 0.5
            self.bullet_client.setJointMotorControl2(self.panda, 12, self.bullet_client.POSITION_CONTROL, spring_link,
                                                     force=100)
            self.bullet_client.setJointMotorControl2(self.panda, 15, self.bullet_client.POSITION_CONTROL, spring_link,
                                                     force=100)


            driver_mimic = amount * 0.8
            self.bullet_client.setJointMotorControl2(self.panda, 10, self.bullet_client.POSITION_CONTROL, driver_mimic,
                                                    force=100)
            self.bullet_client.setJointMotorControl2(self.panda, 13, self.bullet_client.POSITION_CONTROL, driver_mimic,
                                                     force=100)



    def goto_joint_poses(self, jointPoses, gripper):
        indexes = [i for i in range(self.numDofs)]
        index_len = len(indexes)
        local_ll, local_ul = None, None
        # these lower and upper limits were are experimental from moving around the sim. but we don't want them
        # impacting the sim IK
        # local_ll = np.array([-0.36332795, -1.83301728, -2.65733942, -3.04878596, -0.93133401,
        #        1.01175007, -0.66787038, 0.])
        # local_ul = np.array([ 2.96710021,  1.44192887,  0.23807272, -0.5002492,  2.96243465,
        #         3.45266257,  2.40072908,  0.        ])
        if self.arm_type == 'Panda':
            local_ll = np.array([-0.6, -2.2, -3.0, -3.04878596, -np.pi,-np.pi, -np.pi, -np.pi])
            local_ul = np.array([ 3,  1.8, 0.5, -0.5002492,  3.,3.45266257,  2.40072908,  np.pi        ])
            # local_ll = np.array([-np.pi * 2] * self.numDofs)
            # local_ul = np.array([np.pi * 2] * self.numDofs)
            inc = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2])
        elif self.arm_type == 'UR5':
            local_ll = np.array([-np.pi*2]*self.numDofs)
            local_ul = np.array([-0.7,np.pi*2,-0.5,np.pi*2,np.pi*2,np.pi*2])
            inc = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
        targetPoses = np.clip(np.array(jointPoses[:index_len]), local_ll[:index_len], local_ul[:index_len])


        current_poses = np.array([self.bullet_client.getJointState(self.panda, j)[0] for j in range(index_len)])
        # limit the amount the motors can jump in one timestep
        #inc = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        targetPoses = np.clip(targetPoses, current_poses-inc, current_poses+inc)
        # for i in range(self.numDofs):
        #     self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
        #                                              targetPoses[i],
        #                                              force=5 * 240., maxVelocity=0.5)
        self.bullet_client.setJointMotorControlArray(self.panda, indexes, self.bullet_client.POSITION_CONTROL,
                                                     targetPositions=targetPoses ,
                                                     forces=[240.]*len(indexes))

        if gripper is not None:
            self.close_gripper(gripper)

        return targetPoses


    def goto(self, pos=None, orn=None, gripper=None):

        if pos is not None and orn is not None:


            pos = self.add_centering_offset(pos)
            if self.arm_type == 'Panda':
                jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, self.endEffectorIndex, pos, orn, self.ll ,
                                                                           self.ul,
                                                                           self.jr, self.restJointPositions, maxNumIterations=200)
            elif self.arm_type == 'UR5':
                current_poses = np.array([self.bullet_client.getJointState(self.panda, j)[0] for j in range(self.numDofs)])
                # print(current_poses)
                jointPoses = self.IKSolver.calc_angles(pos,orn, current_poses)
                # jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, self.endEffectorIndex, pos, orn, ll,
                #                                                        ul,
                #                                                        jr, self.restJointPositions, maxNumIterations=200)

            targetPoses = self.goto_joint_poses(jointPoses, gripper)
        return targetPoses


    # action must come in as 7 dimensional
    # 3 for pos, 3 for act, 1 for gripper
    def step(self, action):
        assert len(action) == 7
        joints = []


        shift = action[0:3]

        if self.pointMass:
            current_pos = self.bullet_client.getBasePositionAndOrientation(self.mass)[0]
        else:
            current_pos = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[0]
        current_pos = self.subtract_centering_offset(current_pos)

        new_pos = current_pos + shift


        new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
        new_pos[2] = max(new_pos[2], arm_z_min) # z min is very important to stop it going through table
        if self.use_orientation:
            orn_shift = action[3:6]
            current_orn = self.bullet_client.getEulerFromQuaternion(self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[1])
            new_orn = np.array(current_orn) + orn_shift
            new_orn = self.bullet_client.getQuaternionFromEuler(new_orn)
        else:
            new_orn = np.array(self.default_arm_orn)
        if self.fixed_gripper:
            gripper = 0.0
        else:
            gripper = action[-1]

        if self.pointMass:
            self.bullet_client.changeConstraint(self.mass_cid, self.add_centering_offset(new_pos), maxForce=10)
            return None
        else:
            targetPoses = self.goto(new_pos, new_orn, gripper)
            return targetPoses

    # this function is only for fully funcitonal robots, will have ori and gripper
    def absolute_step(self, action):

        assert len(action) == 8
        # Still do relative as far as position goes
        # shift =
        # current_pos = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[0]
        # current_pos = self.subtract_centering_offset(current_pos)
        # new_pos = current_pos + shift

        # all absolute, can do relative predictions on the AI side if we want
        new_pos = action[0:3]
        new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
        #new_pos[2] = max(new_pos[2], arm_z_min)  # z min is very important to stop it going through table

        gripper = action[-1]

        targetPoses = self.goto(new_pos, action[3:7], gripper)
        return targetPoses

        # this function is only for fully funcitonal robots, will have ori and gripper
    def relative_quat_step(self, action):
        assert len(action) == 8

        current_pos = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[0]
        current_orn = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[1]
        new_pos = action[0:3] + current_pos
        new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
        new_orn = action[3:7] + current_orn
        gripper = action[-1]
        targetPoses = self.goto(new_pos, new_orn, gripper)
        return targetPoses

    def absolute_rpy_step(self, action):
        assert len(action) == 7
        new_pos = action[0:3]
        new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
        new_orn = action[3:6]
        gripper = action[-1]
        targetPoses = self.goto(new_pos, self.bullet_client.getQuaternionFromEuler(new_orn), gripper)
        return targetPoses

    def relative_rpy_step(self, action):
        assert len(action) == 7

        current_pos = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[0]
        current_orn = self.bullet_client.getEulerFromQuaternion(self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[1])
        new_pos = action[0:3] + current_pos
        new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
        new_orn = action[3:6] + current_orn
        gripper = action[-1]
        targetPoses = self.goto(new_pos, self.bullet_client.getQuaternionFromEuler(new_orn), gripper)
        return targetPoses

        # take a step with an action commanded in joint space # this doesn't yet have first class support judt trying hey
    def relative_joint_step(self, action):
        current_poses = np.array([self.bullet_client.getJointState(self.panda, j)[0] for j in range(self.numDofs)])
        jointPoses = action[:-1] + current_poses
        gripper = action[-1]
        targetPoses = self.goto_joint_poses(jointPoses, gripper)
        return targetPoses

    def absolute_joint_step(self, action):
        targetPoses = self.goto_joint_poses( action[:-1], action[-1])
        return targetPoses

    def update_state(self):
        keys = self.bullet_client.getKeyboardEvents()

        if len(keys) > 0:
            for k, v in keys.items():
                if v & self.bullet_client.KEY_WAS_TRIGGERED:
                    if (k == ord('1')):
                        self.state = 1
                    if (k == ord('2')):
                        self.state = 2
                    if (k == ord('3')):
                        self.state = 3
                    if (k == ord('4')):
                        self.state = 4
                    if (k == ord('5')):
                        self.state = 5
                    if (k == ord('6')):
                        self.state = 6
                    if (k == ord('8')):
                        self.state = 8

                if v & self.bullet_client.KEY_WAS_RELEASED:
                    self.state = 0

    def human_interactive_step(self, action = None):

        if self.state == 6:
            self.finger_target = 0.01
        if self.state == 5:
            self.finger_target = 0.04

        self.update_state()
        if self.state == 8:
            self.reset()
            return
        #print("self.state=", self.state)
        #print("self.finger_target=", self.finger_target)
        alpha = 0.9  # 0.99
        pos, orn = None, None
        if self.state == 1 or self.state == 2 or self.state == 3 or self.state == 4 or self.state == 7:
            # gripper_height = 0.034
            self.gripper_height = alpha * self.gripper_height + (1. - alpha) * -0.06
            if self.state == 2 or self.state == 3 or self.state == 7:
                self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.1

            t = self.t
            self.t += self.control_dt

            pos = [0.2 * math.sin(1.5 * t), self.gripper_height - self.offset[1], 0.1 * math.cos(1.5 * t)]

            if self.state == 3 or self.state == 4:
                pos, o = self.bullet_client.getBasePositionAndOrientation(self.objects[0])
                pos = self.subtract_centering_offset(pos)
                pos = [pos[0], self.gripper_height - self.offset[1], pos[2]]

                self.prev_pos = pos
            if self.state == 7:
                pos = self.prev_pos
                diffX = pos[0]  # - self.offset[0]
                diffZ = pos[2]  # - (self.offset[2]-0.6)
                self.prev_pos = [self.prev_pos[0] - diffX * 0.1, self.prev_pos[1], self.prev_pos[2] - diffZ * 0.1]

            orn = self.default_arm_orn

        self.goto(pos, orn, self.finger_target)





    def render(self, mode='human'):

        if (mode == "human"):
            self.render_scene = True
            return np.array([])
        if mode == 'rgb_array':
            raise NotImplementedError

    def close(self):
        print('closing')
        self.bullet_client.disconnect()

    def _seed(self, seed=None):
        print('seeding')
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        return [seed]



class pandaEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, num_objects = 0, env_range_low = [-0.18, -0.18,-0.05 ], env_range_high = [0.18, 0.18, 0.15], goal_range_low = [-0.18, -0.18, -0.05], goal_range_high = [0.18, 0.18, 0.05],
                 obj_lower_bound = [-0.18, -0.18, -0.05], obj_upper_bound = [-0.18, -0.18, -0.05], sparse=True, use_orientation=False,
                 sparse_rew_thresh=0.05, pointMass = False, fixed_gripper = False, return_velocity=True, max_episode_steps=250, 
                 play=False, action_type = 'relative', show_goal=True, arm_type= 'Panda'): # action type can be relative, absolute, or joint relative
        fps = 300
        self.timeStep = 1. / fps
        self.render_scene = False
        self.physics_client_active = 0
        self.num_objects  = num_objects
        self.use_orientation = use_orientation
        self.return_velocity = return_velocity
        self.fixed_gripper = fixed_gripper
        self.sparse_reward_threshold = sparse_rew_thresh
        self.num_goals = max(self.num_objects, 1)
        self.play = play
        self.action_type = action_type
        self.show_goal = show_goal
        self.arm_type = arm_type
        obs_dim = 8
        self.sparse_rew_thresh = sparse_rew_thresh
        self._max_episode_steps = max_episode_steps

        obs_dim += 7 * num_objects  # pos and vel of the other pm that we are knocking around.
        # TODO actually clip input actions by this amount!!!!!!!!
        pos_step = 0.015
        orn_step = 0.1
        if self.action_type == 'absolute':
            pos_step = 1.0
            if self.use_orientation:
                high = np.array([pos_step,pos_step,pos_step,1,1,1,1,1]) # use absolute orientations
            else:
                high = np.array([pos_step, pos_step, pos_step, 1])
        elif self.action_type == 'relative_joints':
            if self.arm_type == 'UR5':
                high = np.array([1,1,1,1,1,1, 1])
            else:
                high = np.array([1,1,1,1,1,1,1, 1])
        elif self.action_type == 'absolute_joints':
            if self.arm_type == 'UR5':
                high = np.array([6, 6, 6, 6, 6, 6, 1])
            else:
                high = np.array([6, 6, 6, 6, 6, 6, 6, 1])
        elif self.action_type == 'relative_quat':
            high = np.array([1, 1, 1, 1, 1, 1,1, 1])
        elif self.action_type == 'absolute_rpy':
            high = np.array([6, 6, 6, 6, 6, 6, 1])
        elif self.action_type == 'relative_rpy':
            high = np.array([1,1,1,1,1,1, 1])
        else:
            if self.use_orientation:
                high = np.array([pos_step, pos_step, pos_step, orn_step,orn_step,orn_step, 1])
            else:
                high = np.array([pos_step, pos_step, pos_step, 1])
        self.action_space = spaces.Box(-high, high)
        self.pointMass = pointMass

        self.env_upper_bound = np.array(env_range_high)
        self.env_lower_bound = np.array(env_range_low)
        #self.env_lower_bound[1] = 0  # set the y (updown) min to 0.
        self.goal_upper_bound = np.array(goal_range_high)
        self.goal_lower_bound = np.array(goal_range_low)
        #self.goal_lower_bound[1] = 0  # set the y (updown) min to 0.

        self.obj_lower_bound = obj_lower_bound
        self.obj_upper_bound = obj_upper_bound

        if use_orientation:
            self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1, 0.04])])
            self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1, 0.0])])
            arm_upper_obs_lim = np.concatenate(
                [self.env_upper_bound, np.array([1, 1, 1, 1, 1, 1, 1, 0.04])])  # includes velocity
            arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 1, 1, 1, 1, 0.0])])
            obj_upper_lim = np.concatenate([self.obj_upper_bound, np.ones(7)]) # velocity and orientation
            obj_lower_lim = np.concatenate([self.obj_lower_bound, -np.ones(7)]) # velocity and orientation
            obj_upper_positional_lim = np.concatenate([self.env_upper_bound, np.ones(4)])
            obj_lower_positional_lim = np.concatenate([self.env_lower_bound, -np.ones(4)])
        else:
            self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([0.04])])
            self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([0.0])])
            arm_upper_obs_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 0.04])])  # includes velocity
            arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 0.0])])
            obj_upper_lim = np.concatenate([self.obj_upper_bound, np.ones(3)])
            obj_lower_lim =  np.concatenate([self.obj_lower_bound, -np.ones(3)])
            obj_upper_positional_lim = self.env_upper_bound
            obj_lower_positional_lim = self.env_lower_bound

        upper_obs_dim = np.concatenate([arm_upper_obs_lim] + [obj_upper_lim] * self.num_objects)
        lower_obs_dim = np.concatenate([arm_lower_obs_lim] + [obj_lower_lim] * self.num_objects)
        upper_goal_dim = np.concatenate([self.env_upper_bound] * self.num_goals)
        lower_goal_dim = np.concatenate([self.env_lower_bound] * self.num_goals)

        lower_full_positional_state = np.concatenate([self.arm_lower_lim] + [obj_lower_positional_lim] * self.num_objects) # like the obs dim, but without velocity.
        upper_full_positional_state = np.concatenate([self.arm_upper_lim] + [obj_upper_positional_lim] * self.num_objects)

        #self.action_space = spaces.Box(self.arm_lower_lim, self.arm_upper_lim)

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(lower_goal_dim, upper_goal_dim),
            achieved_goal=spaces.Box(lower_goal_dim, upper_goal_dim),
            observation=spaces.Box(lower_obs_dim, upper_obs_dim),
            controllable_achieved_goal=spaces.Box(self.arm_lower_lim, self.arm_upper_lim),
            full_positional_state=spaces.Box( lower_full_positional_state, upper_full_positional_state)
        ))


        if sparse:
            self.compute_reward = self.compute_reward_sparse



    def reset(self, o = None, vr =None):

        if not self.physics_client_active:
            self.activate_physics_client(vr)
            self.physics_client_active = True

        r = 0
        while r > -1:
            # reset again if we init into a satisfied state

            self.panda.reset(o)
            obs = self.panda.calc_state()
            r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        return obs

    def reset_goal_pos(self, goal):
        self.panda.reset_goal_pos(goal)

    def render(self, mode):
        if (mode == "human"):
            self.render_scene = True
            return np.array([])
        if mode == 'rgb_array':
            raise NotImplementedError
        if mode == 'playback':
            self.panda.record_images = True

    def absolute_command(self, pos, ori):
        ori = p.getQuaternionFromEuler(ori)
        self.panda.goto(pos,ori)
        self.panda.runSimulation()




    def step(self, action= None):
        #bound the action to within allowable limits
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.action_type == 'absolute':
            targetPoses = self.panda.absolute_step(action)
        elif self.action_type == 'relative_joints':
            targetPoses = self.panda.relative_joint_step(action)
        elif self.action_type == 'absolute_joints':
            targetPoses = self.panda.absolute_joint_step(action)
        elif self.action_type == 'relative_quat':
            targetPoses = self.panda.relative_quat_step(action)
        elif self.action_type == 'absolute_rpy':
            targetPoses = self.panda.absolute_rpy_step(action)
        elif self.action_type == 'relative_rpy':
            targetPoses = self.panda.relative_rpy_step(action)
        else:
            targetPoses = self.panda.step(action)

        self.panda.runSimulation()
            # this is out here because the multiprocessing version will step everyone simulataneously.

        # if self.render_scene:
        #     time.sleep(self.timeStep*3)

        obs = self.panda.calc_state()
        r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        done = False
        success = 0 if r < 0 else 1
        for i,o in self.panda.calc_environment_state().items():

            if (o['pos'] > self.env_upper_bound).any() or (o['pos'] < self.env_lower_bound).any():
                done = True
                r = -100



        return obs, r, done, {'is_success': success, 'target_poses': targetPoses}

    def activate_physics_client(self, vr=None):

        if self.render_scene:
            if vr is None:
                self.p = bullet_client.BulletClient(connection_mode=p.GUI)
                #self.p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
                self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            else:
                # Trying to rig up VR
                self.p =  bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)


        #self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        # self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=45, cameraPitch=-90,
        #                              cameraTargetPosition=[0.35, -0.13, 0])
        self.p.setAdditionalSearchPath(pd.getDataPath())

        self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, 0, -9.8)

        if self.play:
            scene = complex_scene
        else:
            if self.num_objects == 0 :
                scene = default_scene
            elif self.num_objects == 1:
                scene = push_scene
        self.panda = pointMassSim(self.p, [0, 0, 0], scene,  self.arm_lower_lim, self.arm_upper_lim,
                                        self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
                                        self.goal_upper_bound, self.obj_lower_bound, self.obj_upper_bound,  self.use_orientation, self.return_velocity,
                                         self.render_scene, pointMass = self.pointMass, fixed_gripper=self.fixed_gripper, 
                                        play=self.play, show_goal = self.show_goal, num_objects=self.num_objects, arm_type=self.arm_type)
        self.panda.control_dt = self.timeStep
        lookat = [0, 0.0, 0.0]
        distance = 0.8
        yaw = 130
        self.p.resetDebugVisualizerCamera(distance, yaw, -130, lookat)

    def vr_activation(self, vr=None):
        #self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        self.p = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY)

        # self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        # self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
        #                              cameraTargetPosition=[0.35, -0.13, 0])
        self.p.setAdditionalSearchPath(pd.getDataPath())

        self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, 0, -9.8)
        scene = complex_scene
        self.panda = pointMassSim(self.p, [0, 0, 0], scene, self.arm_lower_lim, self.arm_upper_lim,
                                  self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
                                  self.goal_upper_bound, self.obj_lower_bound, self.obj_upper_bound,
                                  self.use_orientation, self.return_velocity,
                                  self.render_scene, pointMass=self.pointMass, fixed_gripper=self.fixed_gripper,
                                  play=self.play, num_objects=self.num_objects, arm_type=self.arm_type)
        self.panda.control_dt = self.timeStep
        self.physics_client_active = True





    def calc_target_distance(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance

    def compute_reward(self, achieved_goal, desired_goal):
        return -self.calc_target_distance(achieved_goal,desired_goal)

    def compute_reward_sparse(self, achieved_goal, desired_goal, info=None):

        initially_vectorized = True
        dimension = 3
        if len(achieved_goal.shape) == 1:
            achieved_goal = np.expand_dims(np.array(achieved_goal), axis=0)
            desired_goal = np.expand_dims(np.array(desired_goal), axis=0)
            initially_vectorized = False

        reward = np.zeros(len(achieved_goal))
        # only compute reward on pos not orn for the moment
        g_ag = 0 # increments of dimension, then skip 4 for ori
        g_dg = 0 # incremenets of dimesion
        for g in range(0, self.num_goals):  # piecewise reward
            current_distance = np.linalg.norm(achieved_goal[:, g_ag:g_ag + dimension] - desired_goal[:, g_dg:g_dg + dimension],
                                              axis=1)
            reward += np.where(current_distance > self.sparse_rew_thresh, -1, -current_distance)
            g_ag += dimension+ 4 # for ori
            g_dg += dimension

        if not initially_vectorized:
            return reward[0]
        else:
            return reward

    def activate_human_interactive_mode(self):
        self.panda.step = self.panda.human_interactive_step

    def visualise_sub_goal(self, sub_goal, sub_goal_state = 'full_positional_state'):
        self.panda.visualise_sub_goal(sub_goal, sub_goal_state = sub_goal_state)

    def delete_sub_goal(self):
        self.panda.delete_sub_goal()


class pointMassEnv(pandaEnv):
	def __init__(self, num_objects = 0):
		super().__init__(num_objects=num_objects, use_orientation=False, pointMass = True)

class pandaReach(pandaEnv):
	def __init__(self, num_objects = 0):
		super().__init__(num_objects=num_objects, use_orientation=False)

class pandaPush(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-0.18, -0.18, -0.055], env_range_high = [0.18, 0.18, -0.04],  goal_range_low=[-0.1, -0.1, -0.06], goal_range_high = [0.1, 0.1, -0.05], use_orientation=False): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = goal_range_low, obj_upper_bound = goal_range_high) # in push restrict the inti positions slightly.

class pandaPick(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-0.18, -0.18, -0.055], env_range_high = [0.18, 0.18, 0.2],  goal_range_low=[-0.18, -0.18, 0.0], goal_range_high = [0.18, 0.18, 0.1], use_orientation=False): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = goal_range_low, obj_upper_bound = goal_range_high)

class pandaReach2D(pandaEnv):
	def __init__(self, num_objects = 0, env_range_low = [-0.18, -0.18, -0.07], env_range_high = [0.18, 0.18, 0.0],  goal_range_low=[-0.18, -0.18, -0.06], goal_range_high = [0.18, 0.18, -0.05], use_orientation=False): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high, goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation)

class pandaPlay(pandaEnv):
	def __init__(self, num_objects = 2, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='absolute', show_goal=False)


class pandaPlayRelJoints(pandaEnv):
	def __init__(self, num_objects = 2, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='relative_joints', show_goal=False)

class pandaPlayRelJoints1Obj(pandaEnv):
    def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
        super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                        goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                        obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='relative_joints', show_goal=False)

class pandaPlayAbsJoints1Obj(pandaEnv):
    def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
        super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                        goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                        obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='absolute_joints', show_goal=False)


class pandaPlay1Obj(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='absolute', show_goal=False)

class pandaPlayRel1Obj(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='relative_quat', show_goal=False)

class pandaPlayAbsRPY1Obj(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_rpy', show_goal=False)

class pandaPlayRelRPY1Obj(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                          max_episode_steps=None, play=True, action_type='relative_rpy', show_goal=False)

class UR5PlayAbsRPY1Obj(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_rpy', show_goal=False, arm_type='UR5')

class UR5PlayRelRPY1Obj(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                          max_episode_steps=None, play=True, action_type='relative_rpy', show_goal=False, arm_type='UR5')

class UR5PlayRelJoints1Obj(pandaEnv):
    def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
        super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                        goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                        obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='relative_joints', show_goal=False, arm_type='UR5')

class UR5PlayAbsJoints1Obj(pandaEnv):
    def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
        super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                        goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                        obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_joints', show_goal=False, arm_type='UR5')


class UR5Play1Obj(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute', show_goal=False, arm_type='UR5')

class UR5PlayRel1Obj(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(pointMass = False, num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='relative_quat', show_goal=False, arm_type='UR5')

def add_xyz_rpy_controls(panda):
    controls = []
    orn = panda.panda.default_arm_orn_RPY
    controls.append(panda.p.addUserDebugParameter("X", -1, 1, 0))
    controls.append(panda.p.addUserDebugParameter("Y", -1, 1, 0.00))
    controls.append(panda.p.addUserDebugParameter("Z", -1, 1, 0.2))
    controls.append(panda.p.addUserDebugParameter("R", -4, 4, orn[0]))
    controls.append(panda.p.addUserDebugParameter("P", -4, 4, orn[1]))
    controls.append(panda.p.addUserDebugParameter("Y", -4,4, orn[2]))
    controls.append(panda.p.addUserDebugParameter("grip", panda.action_space.low[-1], panda.action_space.high[-1], 0))
    return controls

def add_joint_controls(panda):

    for i, obj in enumerate(panda.panda.restJointPositions):
        panda.p.addUserDebugParameter(str(i), -2*np.pi, 2*np.pi, obj)






def main():
    joint_control = False #Tru
    panda = UR5PlayAbsRPY1Obj()
    panda.render(mode='human')
    panda.reset()
    if joint_control:

        add_joint_controls(panda)
    else:

        controls = add_xyz_rpy_controls(panda)

    panda.render(mode='human')
    panda.reset()

    for i in range(1000000):

        
    

        if joint_control:
            poses  = []
            for i in range(0, len(panda.panda.restJointPositions)):
                poses.append(panda.p.readUserDebugParameter(i))

            poses[0:len(panda.panda.ul)] = np.clip(poses[0:len(panda.panda.ul)], panda.panda.ll, panda.panda.ul)
            panda.panda.reset_arm_joints(panda.panda.panda, poses)
            print(p.getEulerFromQuaternion(panda.panda.calc_actor_state()['orn']))

        else:
            action = []
            for i in range(0, len(controls)):
                action.append(panda.p.readUserDebugParameter(i))


            #panda.absolute_command(action[0:3], action[3:6])
            state = panda.panda.calc_actor_state()

            #pos_change = action[0:3] - state['pos']
            # des_ori = panda.panda.default_arm_orn #  np.array(action[3:6])
            #des_ori = p.getQuaternionFromEuler(action[3:6])
            # ori_change =  des_ori - np.array(p.getEulerFromQuaternion(np.array(state['orn'])))
            #
            #action = np.concatenate([action[0:3], des_ori, [action[6]]])
            obs, r, done, info = panda.step(np.array(action))
            #print(obs['obs_rpy'][6])
            #print(obs['achieved_goal'][7:])
            #print(p.getEulerFromQuaternion(state['orn']))
            x = obs['achieved_goal']
            x[2] += 0.1
            panda.visualise_sub_goal(obs['achieved_goal'], sub_goal_state='achieved_goal')
            #print(obs['joints'])
            #print(state['pos'], obs['observation'][-4:])

        #time.sleep(0.01)


# def main():
#     panda = pandaPick()
#     controls = []

#     panda.render(mode='human')
#     panda.reset()
#     controls.append(panda.p.addUserDebugParameter("X", panda.action_space.low[0], panda.action_space.high[0], 0))
#     controls.append(panda.p.addUserDebugParameter("Y", panda.action_space.low[1], panda.action_space.high[1], 0.00))
#     controls.append(panda.p.addUserDebugParameter("Z", panda.action_space.low[2], panda.action_space.high[2], 0))
#     controls.append(panda.p.addUserDebugParameter("grip", panda.action_space.low[3], panda.action_space.high[3], 0))
#     state_control = False #True

#     if state_control:
#         panda.activate_human_interactive_mode()

#     for i in range(100000):
#         panda.reset()
#         #time.sleep(10)
#         #panda.visualise_sub_goal(np.array([0,0,0,0.04]), 'controllable_achieved_goal')
#         for j in range(0, 150):

#             action = []
#             if state_control:
#                 panda.step()
#                 time.sleep(0.005)
#             else:
#                 for i in range(0, len(controls)):
#                     action.append(panda.p.readUserDebugParameter(i))
#                 #action = panda.action_space.sample()
#                 obs, r, done, info = panda.step(np.array(action))
#                 #print(obs['achieved_goal'], obs['desired_goal'], r)
#                 print(r)
#                 time.sleep(0.01)


if __name__ == "__main__":
    main()

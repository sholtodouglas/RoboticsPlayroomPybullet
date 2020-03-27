
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

GUI = False
viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=6, yaw=0, pitch=-90, roll=0,
                                                 upAxisIndex=2)

projectionMatrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=10)

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
# upper limits for null space (todo: set them to proper range)
ul = [7] * pandaNumDofs
# joint ranges for null space (todo: set them to proper range)
jr = [7] * pandaNumDofs
# restposes for null space
restJointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = restJointPositions

class pointMassSim():

    def __init__(self, bullet_client, offset, load_scene, arm_lower_lim, arm_upper_lim,
        env_lower_bound, env_upper_bound, goal_lower_bound,
        goal_upper_bound, use_orientation, render_scene, pointMass = False, fixed_gripper = False):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.objects = load_scene(self.bullet_client, offset, flags) # Todo: later, put this after the centering offset so that objects are centered around it too.
        self.num_objects = len(self.objects)
        self.use_orientation = use_orientation
        self.num_objects = len(self.objects)
        self.num_goals = max(self.num_objects, 1)
        self.pointMass = pointMass
        self.arm_lower_lim = arm_lower_lim
        self.arm_upper_lim = arm_upper_lim
        self.env_upper_bound = env_upper_bound
        self.env_lower_bound = env_lower_bound
        self.goal_upper_bound = goal_upper_bound
        self.goal_lower_bound = goal_lower_bound
        self.render_scene = render_scene

        self.physics_client_active = 0
        self.movable_goal = False
        self.roving_goal = False
        self.sub_goals = None


        self.init_arm_orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + offset, self.init_arm_orn,
                                                 useFixedBase=True, flags=flags)

        self.default_arm_orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
        self.original_offset = offset # exclusively for if we span more arms for visulaising sub goals
        self.offset = offset + np.array([0, 0.0, -0.6]) # to center the env about the panda gripper location
        #create a constraint to keep the fingers centered
        self.fixed_gripper = fixed_gripper
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)

        sphereRadius = 0.03
        mass = 1
        colSphereId = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_SPHERE, radius=sphereRadius)
        visId = self.bullet_client.createVisualShape(self.bullet_client.GEOM_SPHERE, radius=sphereRadius,
                                                     rgbaColor=[1, 0, 0, 1])
        if self.pointMass:
            init_loc = self.add_centering_offset(np.array([0, 0.1, 0]))
            self.mass = self.bullet_client.createMultiBody(mass, colSphereId, visId, init_loc)
            collisionFilterGroup = 0
            collisionFilterMask = 0
            self.bullet_client.setCollisionFilterGroupMask(self.mass, -1, collisionFilterGroup,
                                                           collisionFilterMask)


            self.mass_cid = self.bullet_client.createConstraint(self.mass, -1, -1, -1, self.bullet_client.JOINT_FIXED,
                                                                [0, 0, 0], [0, 0, 0],
                                                                init_loc, [0, 0, 0, 1])

            
        self.endEffectorIndex = 11
        self.state = 0
        self.control_dt = 1. / 240.
        self.finger_target = 0
        self.gripper_height = 0.2
        # create the goal objects
        self.show_goal = True
        alpha = 1
        self.obj_colors = [[0, 1, 0, alpha], [0, 0, 1, alpha]]  # colors for two objects
        if self.render_scene and self.show_goal:


            relativeChildOrientation = [0, 0, 0, 1]
            self.goals = []
            self.goal_cids = []
            collisionFilterGroup = 0
            collisionFilterMask = 0
            for g in range(self.num_goals):
                init_loc = self.add_centering_offset(np.array([0, 0.2, 0]))
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
        if obs is None:
            height_offset = 0.03
            for o in self.objects:
                pos = self.add_centering_offset(np.random.uniform(self.env_lower_bound, self.env_upper_bound))
                pos[1] = pos[1] + height_offset  # so they don't collide
                self.bullet_client.resetBasePositionAndOrientation(o, pos, [-0.707107, 0.0, 0.0, 0.707107])
                height_offset += 0.03
            for i in range(0, 20):
                self.bullet_client.stepSimulation()

        else:
            if self.use_orientation:
                index = 8
                increment = 7
            else:
                index = 4
                increment = 3
            for o in self.objects:
                pos = obs[index:index + 3]
                if self.use_orientation:
                    orn = obs[index + 3:index + 7]
                else:
                    orn = [0, 0, 0, 1]
                self.bullet_client.resetBasePositionAndOrientation(o, pos, orn)
                index += increment

    def reset_arm_joints(self, arm, poses):
        index = 0
        for j in range(self.bullet_client.getNumJoints(arm)):
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


    def reset_arm(self, arm, o = None):

        orn = self.default_arm_orn
        if  o is None:
            new_pos = self.add_centering_offset(np.random.uniform(self.env_lower_bound,self.env_upper_bound))
        else:
            new_pos = self.add_centering_offset(o[0:3])
            if self.use_orientation:
                orn = o[3:7]

        if self.pointMass:
            self.bullet_client.resetBasePositionAndOrientation(self.mass,
                                    new_pos, [0,0,0,1])

        self.reset_arm_joints(arm, restJointPositions) # put it into a good init for IK
        # jointPoses = self.bullet_client.calculateInverseKinematics(arm, pandaEndEffectorIndex, new_pos, orn, ll,
        #                                                            ul,
        #                                                            jr, rp, maxNumIterations=20)
        # self.reset_arm_joints(arm, jointPoses)



    def reset(self, o = None):
        if o is not None:
            self.reset_goal_pos(o['desired_goal'])
            self.reset_arm(self.panda, o['observation'])
            self.reset_object_pos(o['observation'])
        else:
            self.reset_goal_pos()
            self.reset_arm(self.panda)
            self.reset_object_pos()
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
            if sub_goal_state is 'full_positional_state' or sub_goal_state is 'controllable_achieved_goal':
                flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                self.ghost_panda = self.bullet_client.loadURDF(
                    os.path.dirname(os.path.abspath(__file__)) + "/franka_panda/ghost_panda.urdf", np.array([0, 0, 0.0]) + self.original_offset,
                    self.init_arm_orn,  useFixedBase = True, flags=flags)

                self.bullet_client.setCollisionFilterGroupMask(self.ghost_panda, -1, collisionFilterGroup,collisionFilterMask)
                for i in range(0, self.bullet_client.getNumJoints(self.ghost_panda)):
                    self.bullet_client.setCollisionFilterGroupMask(self.ghost_panda, i, collisionFilterGroup,
                                                                   collisionFilterMask)

            if sub_goal_state is 'full_positional_state' or sub_goal_state is 'achieved_goal':
                sphereRadius = 0.03
                mass = 1
                colSphereId = self.bullet_client.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
                if sub_goal_state is 'full_positional_state':
                    if self.use_orientation:
                        index = 8
                    else:
                        index = 4
                if sub_goal_state is 'achieved_goal':
                    index = 0
                for i in range(0, self.num_objects):
                    color = self.obj_colors[i]
                    color[3] = 0.5 # set alpha to 0.5 for ghostly subgoal appearance
                    visId = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03],
                                                rgbaColor=color)
                    self.sub_goals.append(self.bullet_client.createMultiBody(mass, colSphereId, visId, sub_goal[index:index+3]))
                    self.bullet_client.setCollisionFilterGroupMask(self.sub_goals[i], -1, collisionFilterGroup,
                                                        collisionFilterMask)
                    index += 3


        if sub_goal_state is 'controllable_achieved_goal':
            self.reset_arm(self.ghost_panda, sub_goal)
        elif sub_goal_state is 'full_positional_state':
            self.reset_arm(self.ghost_panda, sub_goal)
            if self.use_orientation:
                index = 8
            else:
                index = 4
        elif sub_goal_state is 'achieved_goal':
            index = 0
        if sub_goal_state is not 'controllable_achieved_goal':
            for i in range(0, self.num_objects):
                print('----', sub_goal[index:index+3])
                self.bullet_client.resetBasePositionAndOrientation(self.sub_goals[i], self.add_centering_offset(sub_goal[index:index+3]), [0,0,0,1])
                index += 3


    def calc_actor_state(self):
        state = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)
        pos, orn, vel, orn_vel = state[0], state[1], state[-2], state[-1]

        gripper_state = [self.bullet_client.getJointState(self.panda, 9)[0]]
        if self.pointMass:
            pos = self.bullet_client.getBasePositionAndOrientation(self.mass)[0]
            vel = self.bullet_client.getBaseVelocity(self.mass)[0]


        return {'pos': self.subtract_centering_offset(pos), 'orn': orn, 'pos_vel': vel, 'orn_vel': orn_vel,
                'gripper': gripper_state}


    def calc_environment_state(self):
        object_states = {}
        for i in range(0, self.num_objects):
            pos, orn = self.bullet_client.getBasePositionAndOrientation(self.objects[i])
            object_states[i] = {'pos': self.subtract_centering_offset(pos), 'orn': orn}

        return object_states

    def calc_state(self):

        arm_state = self.calc_actor_state()
        if self.use_orientation:
            arm_elements = ['pos', 'pos_vel', 'orn', 'gripper']
        else:
            arm_elements = ['pos', 'pos_vel', 'gripper']
        state = np.concatenate([np.array(arm_state[i]) for i in arm_elements])
        if self.num_objects > 0:
            env_state = self.calc_environment_state()

            if self.use_orientation:
                obj_states = np.concatenate(
                    [np.concatenate([obj['pos'], obj['orn']]) for (i, obj) in env_state.items()])
            else:
                obj_states = np.concatenate([obj['pos'] for (i, obj) in env_state.items()])
            state = np.concatenate([state, obj_states])
            achieved_goal = np.concatenate([obj['pos'] for (i, obj) in env_state.items()]) # todo ag orientation?
            full_positional_state = np.concatenate([arm_state['pos'], achieved_goal])
        else:
            achieved_goal = arm_state['pos']
            full_positional_state = achieved_goal

        return_dict = {
            'observation': state.copy().astype('float32'),
            'achieved_goal': achieved_goal.copy().astype('float32'),
            'desired_goal': self.goal.copy().astype('float32'),
            'controllable_achieved_goal': np.concatenate([arm_state['pos'].copy(), arm_state['gripper'].copy()]).astype('float32'),
            # just the x,y pos of the pointmass, the controllable aspects
            'full_positional_state': full_positional_state.copy().astype('float32')
        }


        return return_dict

    def goto(self, pos=None, orn=None, gripper=None):

        if pos is not None and orn is not None:


            pos = self.add_centering_offset(pos)
            jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll,
                                                                       ul,
                                                                       jr, rp, maxNumIterations=20)

            indexes = [i for i in range(pandaNumDofs)]
            # for i in range(pandaNumDofs):
            #     self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
            #                                              jointPoses[i],
            #                                              force=5 * 240.)
            self.bullet_client.setJointMotorControlArray(self.panda, indexes, self.bullet_client.POSITION_CONTROL,
                                                         targetPositions= jointPoses[:len(indexes)],
                                                         forces=[5 * 240.]*len(indexes))



        if gripper is not None:

            for i in [9, 10]:
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, gripper,
                                                         force=10)


    def step(self, action):
        shift = action[0:3]

        if self.pointMass:
            current_pos = self.bullet_client.getBasePositionAndOrientation(self.mass)[0]
        else:
            current_pos = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[0]
        current_pos = self.subtract_centering_offset(current_pos)

        new_pos = current_pos + shift

        new_pos = np.clip(new_pos, self.env_lower_bound, self.env_upper_bound)
        if self.use_orientation:
            orn_shift = np.array(action[3:7])
            current_orn = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)[1]
            new_orn = current_orn + orn_shift
            new_orn = np.clip(new_orn, -np.ones(4), np.ones(4))
        else:
            new_orn = np.array(self.default_arm_orn)

        if self.fixed_gripper:
            gripper = 0.0
        else:
            gripper = action[-1]
        self.goto(new_pos, new_orn, gripper)
        if self.pointMass:
            self.bullet_client.changeConstraint(self.mass_cid, self.add_centering_offset(new_pos), maxForce=100)


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





    def render(self, mode):

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

    def __init__(self, num_objects = 0, env_range_low = [-0.2, -0.1, -0.2], env_range_high = [0.2, 0.1, 0.2], goal_range_low = [-0.2, -0.1, -0.2], goal_range_high = [0.2, 0.1, 0.2],
                 sparse=True, use_orientation=False, sparse_rew_thresh=0.05, pointMass = False, fixed_gripper = False):
        fps = 240
        self.timeStep = 1. / fps
        self.render_scene = False
        self.physics_client_active = 0
        self.num_objects  = num_objects
        self.use_orientation = use_orientation
        self.fixed_gripper = fixed_gripper
        self.sparse_reward_threshold = sparse_rew_thresh
        self.num_goals = max(self.num_objects, 1)


        obs_dim = 7
        self.sparse_rew_thresh = sparse_rew_thresh
        self._max_episode_steps = 250

        obs_dim += 6 * num_objects  # pos and vel of the other pm that we are knocking around.
        pos_step = 0.03
        orn_step = 0.05
        if self.use_orientation:
            high = np.array([pos_step, pos_step, pos_step, orn_step,orn_step,orn_step,orn_step, 0.04])
        else:
            high = np.array([pos_step, pos_step, pos_step, 0.04])
        self.action_space = spaces.Box(-high, high) # interesting how key this is. Oh because you know what, before it couldn't fucking go down. Duh.
        self.pointMass = pointMass

        self.env_upper_bound = np.array(env_range_high)
        self.env_lower_bound = np.array(env_range_low)
        #self.env_lower_bound[1] = 0  # set the y (updown) min to 0.
        self.goal_upper_bound = np.array(goal_range_high)
        self.goal_lower_bound = np.array(goal_range_low)
        #self.goal_lower_bound[1] = 0  # set the y (updown) min to 0.


        if use_orientation:
            self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1, 0.04])])
            self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1, 0.0])])
            arm_upper_obs_lim = np.concatenate(
                [self.env_upper_bound, np.array([1, 1, 1, 1, 1, 1, 1, 0.04])])  # includes velocity
            arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 1, 1, 1, 1, 0.0])])
            obj_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1])])
            obj_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1])])
        else:
            self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([0.04])])
            self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([0.0])])
            arm_upper_obs_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 0.04])])  # includes velocity
            arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 0.0])])
            obj_upper_lim = self.env_upper_bound
            obj_lower_lim = self.env_lower_bound

        upper_obs_dim = np.concatenate([arm_upper_obs_lim] + [obj_upper_lim] * self.num_objects)
        lower_obs_dim = np.concatenate([arm_lower_obs_lim] + [obj_lower_lim] * self.num_objects)
        upper_goal_dim = np.concatenate([self.env_upper_bound] * self.num_goals)
        lower_goal_dim = np.concatenate([self.env_lower_bound] * self.num_goals)

        lower_full_positional_state = np.concatenate([self.arm_lower_lim] + [obj_lower_lim] * self.num_objects) # like the obs dim, but without velocity.
        upper_full_positional_state = np.concatenate([self.arm_upper_lim] + [obj_lower_lim] * self.num_objects)

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



    def reset(self):

        if not self.physics_client_active:
            self.activate_physics_client()
            self.physics_client_active = True

        r = 0
        while r > -1:
            # reset again if we init into a satisfied state
            self.panda.reset()
            obs = self.panda.calc_state()
            r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
            print('resetty', r)
            if r > -1:
                print('break and reset')

        return obs

    def render(self, mode):
        if (mode == "human"):
            self.render_scene = True
            return np.array([])
        if mode == 'rgb_array':
            raise NotImplementedError



    def step(self, action= None):

        self.panda.step(action)

        for i in range(0, 20):
            self.p.stepSimulation()
            # this is out here because the multiprocessing version will step everyone simulataneously.

        if self.render_scene:
            time.sleep(self.timeStep*3)

        obs = self.panda.calc_state()
        r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        done = False
        success = 0 if r < 0 else 1
        # TODO: Should it be done if we exceed the environment limits to prevent bad data?
        return obs, r, done, {'is_success': success}

    def activate_physics_client(self):

        if self.render_scene:

            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
            self.p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
            self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)


        #self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        # self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
        #                              cameraTargetPosition=[0.35, -0.13, 0])
        self.p.setAdditionalSearchPath(pd.getDataPath())

        self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, -9.8, 0)

        if self.num_objects == 0 :
            scene = default_scene
        elif self.num_objects == 1:
            scene = push_scene
        elif self.num_objects == 2:
            scene = complex_scene
        self.panda = pointMassSim(self.p, [0, 0, 0], scene,  self.arm_lower_lim, self.arm_upper_lim,
                                        self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
                                        self.goal_upper_bound,  self.use_orientation, self.render_scene, pointMass = self.pointMass, fixed_gripper=self.fixed_gripper)
        self.panda.control_dt = self.timeStep
        lookat = [0, 0, 0.0]
        distance = 1.3
        yaw = 140
        self.p.resetDebugVisualizerCamera(distance, yaw, -150, lookat)




    def calc_target_distance(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance

    def compute_reward(self, achieved_goal, desired_goal):
        return -self.calc_target_distance(achieved_goal,desired_goal)


    def compute_reward_sparse(self, achieved_goal, desired_goal, info=None):
        # I know this should be vectorized but cbf atm - my HER doesn't use vector ag and dg. Just need this for baselines
        # compatability.
        # This implements a piecewise reward function, for multiobject envs.
        initially_vectorized = True
        dimensions = 3
        if len(achieved_goal.shape) == 1:
            achieved_goal = np.expand_dims(np.array(achieved_goal), axis=0)
            desired_goal = np.expand_dims(np.array(desired_goal), axis=0)
            initially_vectorized = False
        reward = np.zeros(len(achieved_goal))
        for i in range(0, len(achieved_goal)):
            for g in range(0, len(achieved_goal[i]) // dimensions ): # three for three dimensions
                g = g * dimensions   # increments of 2
                current_distance = self.calc_target_distance(achieved_goal[g:g + dimensions ], desired_goal[g:g + dimensions ])
                if current_distance > self.sparse_reward_threshold:
                    reward[i] -= 1
                if current_distance < self.sparse_reward_threshold:
                    reward[i] -= current_distance # semi sparse reward, rewards centering it exactly!

        if not initially_vectorized:
            return reward[0]
        else:
            return reward

    def activate_human_interactive_mode(self):
        self.panda.step = self.panda.human_interactive_step

    def visualise_sub_goal(self, sub_goal, sub_goal_state = 'full_positional_state'):
        self.panda.visualise_sub_goal(sub_goal, sub_goal_state = sub_goal_state)


class pointMassEnv(pandaEnv):
	def __init__(self, num_objects = 0):
		super().__init__(num_objects=num_objects, use_orientation=False, pointMass = True)

class pandaReach(pandaEnv):
	def __init__(self, num_objects = 0):
		super().__init__(num_objects=num_objects, use_orientation=False)

class pandaPush(pandaEnv):
	def __init__(self, num_objects = 1, env_range_low = [-0.15, -0.055, -0.15], env_range_high = [0.15, -0.04, 0.15],  goal_range_low=[-0.1, -0.06, -0.1], goal_range_high = [0.1, -0.05, 0.1], use_orientation=False): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high, goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation)


class pandaReach2D(pandaEnv):
	def __init__(self, num_objects = 0, env_range_low = [-0.2, -0.07, -0.2], env_range_high = [0.2, 0.0, 0.2],  goal_range_low=[-0.2, -0.06, -0.2], goal_range_high = [0.2, -0.05, 0.2], use_orientation=False): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high, goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation)


def main():
    panda = pandaPush()
    controls = []

    panda.render(mode='human')
    panda.reset()
    controls.append(panda.p.addUserDebugParameter("X", panda.action_space.low[0], panda.action_space.high[0], 0))
    controls.append(panda.p.addUserDebugParameter("Y", panda.action_space.low[1], panda.action_space.high[1], 0.02))
    controls.append(panda.p.addUserDebugParameter("Z", panda.action_space.low[2], panda.action_space.high[2], 0))
    controls.append(panda.p.addUserDebugParameter("grip", panda.action_space.low[3], panda.action_space.high[3], 0))
    state_control = False #True

    if state_control:
        panda.activate_human_interactive_mode()

    for i in range(100000):
        panda.reset()
        #panda.visualise_sub_goal(np.array([0,0,0,0.04]), 'controllable_achieved_goal')
        for j in range(0, 5):

            action = []
            if state_control:
                panda.step()
                time.sleep(0.005)
            else:
                for i in range(0, len(controls)):
                    action.append(panda.p.readUserDebugParameter(i))
                #action = panda.action_space.sample()
                obs, r, done, info = panda.step(np.array(action))
                #print(obs['achieved_goal'], obs['desired_goal'], r)
                print(r)
                time.sleep(0.01)


if __name__ == "__main__":
    main()
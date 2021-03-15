
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
from playRewardFunc import success_func
from inverseKinematics import InverseKinematicsSolver

lookat = [0, 0.0, 0.0]
distance = 0.8
yaw = 130
pitch = -130
pixels = 200
viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=lookat, distance=distance, yaw=yaw, pitch=pitch, roll=0,
                                                 upAxisIndex=2)
projectionMatrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=10)

# A function used to get camera images from the gripper - experimental
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

# An instance of the environment. Multiples of these can be placed in the same 
# 'env' by using the offset parameter - and obs/acts will be placed into the offsetby the
# add/subtract centering offset functions. 

'''
The main gym env.
This was originally set up to support RL, so that you could have many many instances (1 arm + tabltop env) at different positions in the 
world (using the 'offset' arg each time you created one). It currently only supports one instance, but would be easy to adapt to multiple instances
with loops inside step, activate physics client, and compute reward functions
'''
class playEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, num_objects = 0, env_range_low = [-0.18, -0.18,-0.05 ], env_range_high = [0.18, 0.18, 0.15], goal_range_low = [-0.18, -0.18, -0.05], goal_range_high = [0.18, 0.18, 0.05],
                 obj_lower_bound = [-0.18, -0.18, -0.05], obj_upper_bound = [-0.18, -0.18, -0.05], sparse=True, use_orientation=False,
                 sparse_rew_thresh=0.05, fixed_gripper = False, return_velocity=True, max_episode_steps=250, 
                 play=False, action_type = 'absolute_rpy', show_goal=True, arm_type= 'Panda'): # action type can be relative, absolute, or joint relative
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
        if action_type == 'absolute_quat':
            pos_step = 1.0
            if self.use_orientation:
                high = np.array([pos_step,pos_step,pos_step,1,1,1,1,1]) # use absolute orientations
            else:
                high = np.array([pos_step, pos_step, pos_step, 1])
        elif action_type == 'relative_quat':
            high = np.array([1, 1, 1, 1, 1, 1,1, 1])
        elif action_type == 'relative_joints':
            if self.arm_type == 'UR5':
                high = np.array([1,1,1,1,1,1, 1])
            else:
                high = np.array([1,1,1,1,1,1,1, 1])
        elif action_type == 'absolute_joints':
            if self.arm_type == 'UR5':
                high = np.array([6, 6, 6, 6, 6, 6, 1])
            else:
                high = np.array([6, 6, 6, 6, 6, 6, 6, 1])
        elif action_type == 'absolute_rpy':
            high = np.array([6, 6, 6, 6, 6, 6, 1])
        elif action_type == 'relative_rpy':
            high = np.array([1,1,1,1,1,1, 1])
        else:
            if self.use_orientation:
                high = np.array([pos_step, pos_step, pos_step, orn_step,orn_step,orn_step, 1])
            else:
                high = np.array([pos_step, pos_step, pos_step, 1])
        self.action_space = spaces.Box(-high, high)
        

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

    # Resets the instances until reward is not satisfied
    def reset(self, o = None, vr =None):

        if not self.physics_client_active:
            self.activate_physics_client(vr)
            self.physics_client_active = True

        r = 0
        while r > -1:
            # reset again if we init into a satisfied state

            self.instance.reset(o)
            obs = self.instance.calc_state()
            r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        return obs

    # Directly reset the goal position
    def reset_goal_pos(self, goal):
        self.instance.reset_goal_pos(goal)

    # Sets whether to render the scene
    # Img will be returned as part of the state which comes from obs, 
    # rather than from calling .render(rgb) every time as in some other envs
    def render(self, mode):
        if (mode == "human"):
            self.render_scene = True
            return np.array([])
        if mode == 'rgb_array':
            self.instance.record_images = True
        if mode == 'playback':
            self.instance.record_images = True

    # Classic dict-env .step function
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        targetPoses = self.instance.perform_action(action, self.action_type)
        self.instance.runSimulation()
        obs = self.instance.calc_state()
        r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        done = False
        success = 0 if r < 0 else 1
        return obs, r, done, {'is_success': success, 'target_poses': targetPoses}

    # Activates the GUI or headless physics client, and creates arm instance within it
    # Within this function is the call which selects which scene 'no obj, one obj, play etc - defined in scenes.py'
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
        self.instance = instance(self.p, [0, 0, 0], scene,  self.arm_lower_lim, self.arm_upper_lim,
                                        self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
                                        self.goal_upper_bound, self.obj_lower_bound, self.obj_upper_bound,  self.use_orientation, self.return_velocity,
                                         self.render_scene, fixed_gripper=self.fixed_gripper, 
                                        play=self.play, show_goal = self.show_goal, num_objects=self.num_objects, arm_type=self.arm_type)
        self.instance.control_dt = self.timeStep
        self.p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    # With VR, a physics server is already created - this connects to it
    def vr_activation(self, vr=None):
        self.p = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY)

        self.p.setAdditionalSearchPath(pd.getDataPath())

        self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, 0, -9.8)
        scene = complex_scene
        self.instance = instance(self.p, [0, 0, 0], scene, self.arm_lower_lim, self.arm_upper_lim,
                                  self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
                                  self.goal_upper_bound, self.obj_lower_bound, self.obj_upper_bound,
                                  self.use_orientation, self.return_velocity,
                                  self.render_scene, fixed_gripper=self.fixed_gripper,
                                  play=self.play, num_objects=self.num_objects, arm_type=self.arm_type)
        self.instance.control_dt = self.timeStep
        self.physics_client_active = True

    def calc_target_distance(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance

    # A basic dense reward metric, distance from goal state
    def compute_reward(self, achieved_goal, desired_goal):
        return -self.calc_target_distance(achieved_goal,desired_goal)

    # A piecewise reward function, for each object it determines if all dims are within the threshold and increments reward if so
    def compute_reward_sparse(self, achieved_goal, desired_goal, info=None):
        if self.play:
            # This success function lives in 'playRewardFunc.py'
            return success_func(achieved_goal, desired_goal)
        else:
            initially_vectorized = True
            dimension = 3
            if len(achieved_goal.shape) == 1:
                achieved_goal = np.expand_dims(np.array(achieved_goal), axis=0)
                desired_goal = np.expand_dims(np.array(desired_goal), axis=0)
                initially_vectorized = False

            reward = np.zeros(len(achieved_goal))
            # only compute reward on pos not orn for the moment
            g_ag = 0 # increments of dimension, then skip 4 for ori
            g_dg = 0 # increments of dimension,
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

    # Env level sub goal visualisation and deletion 
    def visualise_sub_goal(self, sub_goal, sub_goal_state = 'full_positional_state'):
        self.instance.visualise_sub_goal(sub_goal, sub_goal_state = sub_goal_state)

    def delete_sub_goal(self):
        try:
            self.instance.delete_sub_goal()
        except:
            pass


class instance():
    def __init__(self, bullet_client, offset, load_scene, arm_lower_lim, arm_upper_lim,
        env_lower_bound, env_upper_bound, goal_lower_bound,
        goal_upper_bound, obj_lower_bound, obj_upper_bound, use_orientation, return_velocity, render_scene,
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
            self.init_arm_base_orn = p.getQuaternionFromEuler([0, 0, 0])
            self.endEffectorIndex = 11
            self.restJointPositions = [-0.6, 0.437, 0.217, -2.09, 1.1, 1.4, 1.3, 0.0, 0.0, 0.0]
            self.numDofs = 7
            self.init_arm_base_pos =  np.array([-0.5, 0.0, -0.05])
        elif self.arm_type == 'UR5':
            self.default_arm_orn_RPY = [0, 0, 0]
            self.default_arm_orn = self.bullet_client.getQuaternionFromEuler(self.default_arm_orn_RPY)
            self.init_arm_base_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])
            self.endEffectorIndex = 7
            # this pose makes it very easy for the IK to do the 'underhand' grip, which isn't well solved for
            # if we take an over hand top down as our default (its very easy for it to flip into an unideal configuration)
            self.restJointPositions =  [-1.50189075, - 1.6291067, - 1.87020409, - 1.21324173, 1.57003561, 0.06970189]
            self.numDofs = 6
            self.init_arm_base_pos = np.array([0.5, -0.1, 0.0])

        else:
            raise NotImplementedError
        self.ll = [-7] * self.numDofs
        self.ul = [7] * self.numDofs
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
        
        print(currentdir)
        global ll
        global ul
        if self.arm_type == 'Panda':

            self.arm = self.bullet_client.loadURDF(currentdir + "/franka_panda/panda.urdf",
                                                        self.init_arm_base_pos + offset,
                                                        self.init_arm_base_orn , useFixedBase=True, flags=flags)
            c = self.bullet_client.createConstraint(self.arm,9,self.arm,10,
                                                    jointType=self.bullet_client.JOINT_GEAR,
                                                    jointAxis=[1, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
            self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        elif self.arm_type == 'UR5':

            self.arm = self.bullet_client.loadURDF(currentdir + "/ur_e_description/ur5e2.urdf",
                                                        self.init_arm_base_pos + offset,
                                                        self.init_arm_base_orn, useFixedBase=True, flags=flags)
            self.IKSolver  = InverseKinematicsSolver( self.init_arm_base_pos, self.init_arm_base_orn, self.endEffectorIndex, self.restJointPositions)


        else:
            raise NotImplementedError

        self.offset = offset + centering_offset  # to center the env about the gripper location
        # create a constraint to keep the fingers centered

        for j in range(self.bullet_client.getNumJoints(self.arm)):
            self.bullet_client.changeDynamics(self.arm, j, linearDamping=0, angularDamping=0)


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

    # Adds the offset (i.e to actions or objects being created) so that all instances share an action/obs space
    def add_centering_offset(self, numbers):
        # numbers either come in as xyz, or xyzxyzxyz etc (in the case of goals or achieved goals)
        offset = np.array(list(self.offset) * (len(numbers) // 3))
        numbers = numbers + offset
        return numbers
    # Subtract the offset (i.e when reading out states) so the obs observed have correct info
    def subtract_centering_offset(self, numbers):
        offset = np.array(list(self.offset) * (len(numbers) // 3))
        numbers = numbers - offset
        return numbers

    # Checks if the button or dial was pressed, and changes the environment to reflect it
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
    # Takes environment steps s.t the sim runs at 25 Hz
    def runSimulation(self):
        # also do toggle updating here
        if self.play:
            self.updateToggles() # so its got both in VR and replay out
        for i in range(0, 12):  # 25Hz control at 300
            self.bullet_client.stepSimulation()
    # Resets goal positions, if a goal is passed in - it will reset the goal to that position
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
    # Resets object positions, if an obs is passed in - the objects will be reset using that
    def reset_object_pos(self, obs=None):
        # Todo object velocities to make this properly deterministic
        if self.play:
            self.bullet_client.resetBasePositionAndOrientation(self.drawer['drawer'], self.drawer['defaults']['pos'],
                                                               self.drawer['defaults']['ori'])
            for i in self.joints:
                self.bullet_client.resetJointState(i, 0, 0) # reset door, button etc

        if obs is None:
            height_offset = 0.03
            for o in self.objects:
                pos = self.add_centering_offset(np.random.uniform(self.obj_lower_bound, self.obj_upper_bound))
                pos[2] = pos[2] + height_offset  # so they don't collide
                self.bullet_client.resetBasePositionAndOrientation(o, pos, [0.0, 0.0, 0.7071, 0.7071])
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
    # Resets the arm joints to a specific pose
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
    # Resets the arm - if o is specified it will reset to that pose
    # 'Which arm' specifies whether it is the actual arm, or the ghostly arm which can be used to visualise sub-goals in hierarchial settings (only implemented for Panda)
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

        jointPoses = self.bullet_client.calculateInverseKinematics(which_arm, self.endEffectorIndex, new_pos, orn,)[0:6]


        self.reset_arm_joints(which_arm, jointPoses)

    # Overall reset function, passes o to the above specific reset functions if sepecified
    def reset(self, o = None):
        self.reset_object_pos(o)
        self.reset_arm(self.arm, o)
        self.reset_goal_pos()
        self.t = 0

    # Visualises the sub-goal passed to it as transparent version of the current goals (whether environment pieces, or the arm itself in reaching tasks)
    def visualise_sub_goal(self, sub_goal, sub_goal_state = 'achieved_goal'): 
        '''
        Supports a number of different types of goal, all of which are returned by the 'calc_state' function
        Full positional state : This is every (non velocity) aspect of the obs as a goal
        Controllable achieved goal : Only the aspects of the env that are controllable, i.e its own pos/ori, not object
        Achieved goal : Only the non controllable aspects of the env, i.e objects, not its own pos/ori
        '''
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
                #print(index+i)
                self.bullet_client.resetJointState(j, 0, sub_goal[index+i])  # reset drawer, button etc

    # Deletes the sub-goal viz, as pyBullet's state reset does not work if extra objects are in the scene 
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

    # Binary return indicating if something is between the gripper prongs, currently unused. 
    def gripper_proprioception(self):
        if self.arm_type == 'UR5':
            gripper_one = np.array(self.bullet_client.getLinkState(self.arm, 18)[0])
            gripper_two = np.array(self.bullet_client.getLinkState(self.arm, 20)[0])
            ee = np.array(self.bullet_client.getLinkState(self.arm, self.endEffectorIndex)[0])
            wrist = np.array(self.bullet_client.getLinkState(self.arm, self.endEffectorIndex - 1)[0])
            avg_gripper = (gripper_one + gripper_two) / 2
            point_one = ee - (ee - wrist) * 0.5 # far up
            point_two = avg_gripper + (ee - wrist) * 0.2 # between the prongs

            try:
                obj_id, link_index, hit_fraction, hit_position, hit_normal = self.bullet_client.rayTest(point_one, point_two)[0]
                
                # visualises the ray getting tripped
                #self.bullet_client.addUserDebugLine(gripper_one, gripper_two, [1,0,0], 0.5, 1)
                
                if hit_fraction == 1.0 or link_index == 18 or link_index == 20:
                    return 0 # nothing in the hand
                else:
                    return 1 # something in the way, oooh, something in the way
            except:
                return -1 # this shouldn't ever happen because the ray will always hit the other side of the gripper
        else:
            return -1

    # Calculates the state of the arm
    def calc_actor_state(self):
        '''
        Returns a dict of all the pieces of information you could want (e.g pos, orn, vel, orn vel, gripper, joints)
        '''
        state = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)
        pos, orn, vel, orn_vel = state[0], state[1], state[-2], state[-1]

        if self.arm_type == 'Panda':
            gripper_state = [self.bullet_client.getJointState(self.arm, 9)[0]]
        else:
            gripper_state = [self.bullet_client.getJointState(self.arm, 18)[0]*23] # put it on a 0-1 scale

        joint_poses = [self.bullet_client.getJointState(self.arm, j)[0] for j in range(8)]

        # If you want, you can access a gripper cam like this - you'll need to actually return it in the dict though
        #img = gripper_camera(self.bullet_client, pos, orn)
        
        return {'pos': self.subtract_centering_offset(pos), 'orn': orn, 'pos_vel': vel, 'orn_vel': orn_vel,
                'gripper': gripper_state, 'joints':joint_poses, 'proprioception': self.gripper_proprioception()}

    # Calculates the state of the environment
    def calc_environment_state(self):
        '''
        First gets the pos (xyz) and orn (q1-4) of each object in the scene, then gets play specific objects
        in order drawer, door, button, dial. E.g, a 1 object scene will have an 11D return from this
        Returns it as a dict of dicts, each one representing one object or 1D return in the environment
        '''
        object_states = {}
        for i in range(0, self.num_objects):
            pos, orn = self.bullet_client.getBasePositionAndOrientation(self.objects[i])
            vel = self.bullet_client.getBaseVelocity(self.objects[i])[0]
            object_states[i] = {'pos': self.subtract_centering_offset(pos), 'orn': orn, 'vel':vel}

        # get things like hinges, doors, dials, buttons etc
        
        if self.play:
            i += 1
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

    
    # Combines actor and environment state into vectors, and takes all the different slices one could want in a return dict
    # Vector size is different depending on whether you are returning just pos, pos & orn, pos, orn & vel etc as specified
    # Keys to know :  observation (full state, no vel), achieved_goal (just environment state), desired_goal (currently specified goal)
    def calc_state(self):

        if self.play:
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
            img_arr = self.bullet_client.getCameraImage(pixels, pixels, viewMatrix, projectionMatrix, flags=self.bullet_client.ER_NO_SEGMENTATION_MASK, shadow=0,
                                       renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)[2][:,:,:3] #just the rgb
        else:
            img_arr = None

        if self.play:
            state, achieved_goal = self.quaternion_safe_the_obs(state, achieved_goal)
        return_dict = {
            'obs_quat': state.copy().astype('float32'),
            'achieved_goal': achieved_goal.copy().astype('float32'),
            'desired_goal': self.goal.copy().astype('float32'),
            'controllable_achieved_goal': np.concatenate([arm_state['pos'].copy(), arm_state['gripper'].copy()]).astype('float32'),
            # just the x,y,z pos of the self, the controllable aspects
            'full_positional_state': full_positional_state.copy().astype('float32'),
            'joints': arm_state['joints'],
            'velocity': np.concatenate([arm_state['pos_vel'], arm_state['orn_vel']]),
            'img': img_arr,
            'observation': np.concatenate([state[0:3], p.getEulerFromQuaternion(state[3:7]), state[7:]]).copy(),
            'gripper_proprioception':  arm_state['proprioception']
        }


        return return_dict

    # PyBullet randomly returns equivalent quaternions (e.g, one timestep it will return the -ve of the previous quat)
    # For a smooth state signal, flip the quaternion if all elements are -ive of prev step
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
    
    # Step 1.  Understand what type of action was commanded
    def perform_action(self, action, action_type):
        '''
        Takes in the action, and uses the appropriate function to determie to get joint angles
        and perform the action in the environment
        '''
        if action_type == 'absolute_quat':
            targetPoses = self.absolute_quat_step(action)
        elif action_type == 'relative_quat':
            targetPoses = self.relative_quat_step(action)
        elif action_type == 'relative_joints':
            targetPoses = self.relative_joint_step(action)
        elif action_type == 'absolute_joints':
            targetPoses = self.absolute_joint_step(action)
        elif action_type == 'absolute_rpy':
            targetPoses = self.absolute_rpy_step(action)
        elif action_type == 'relative_rpy':
            targetPoses = self.relative_rpy_step(action)
        else:
            raise NotImplementedError
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
        new_orn = action[3:7] + current_orn
        gripper = action[-1]
        targetPoses =  self.goto(new_pos, new_orn, gripper)
        return targetPoses
    def absolute_rpy_step(self, action):
        assert len(action) == 7
        new_pos = action[0:3]
        new_orn = action[3:6]
        gripper = action[-1]
        targetPoses =  self.goto(new_pos, self.bullet_client.getQuaternionFromEuler(new_orn), gripper)
        return targetPoses
    def relative_rpy_step(self, action):
        assert len(action) == 7
        current_pos = self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[0]
        current_orn = self.bullet_client.getEulerFromQuaternion(self.bullet_client.getLinkState(self.arm, self.endEffectorIndex, computeLinkVelocity=1)[1])
        new_pos = action[0:3] + current_pos
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
        targetPoses = self.goto_joint_poses(action[:-1], action[-1])
        return targetPoses

    # Step 1.5. Convert these to xyz quat if they are not joints
    def goto(self, pos=None, orn=None, gripper=None):
        '''
        Uses PyBullet IK to solve for desired joint angles
        We use a background, headless self to stabilise IK predictions for the UR5, the extra few cm of accuracy is worth the 
        neglible slowdown
        '''
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

    # Step 2. Send joint commands to the robot - by default UR5 uses a support function in 'inverseKinematics.py' for greater stability
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

    # Command to control the gripper from 0-1
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

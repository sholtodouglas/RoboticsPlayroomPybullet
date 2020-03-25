import gym, gym.utils, gym.utils.seeding
import pybullet as p
import numpy as np
import pybullet_data
import os
import time
from pybullet_utils import bullet_client

urdfRoot = pybullet_data.getDataPath()
import gym.spaces as spaces
import math

GUI = False
viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=6, yaw=0, pitch=-90, roll=0,
                                                 upAxisIndex=2)

projectionMatrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=10)


class pointMassEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, render=False, num_objects=0, sparse=True, TARG_LIMIT=2, sparse_rew_thresh=0.3):
        self.num_objects = num_objects
        self.use_orientation = False
        action_dim = 4
        obs_dim = 7
        self.ENVIRONMENT_BOUNDS = 2.5  # LENGTH 6
        self.sparse_rew_thresh = sparse_rew_thresh
        self._max_episode_steps = 250

        obs_dim += 6 * num_objects  # pos and vel of the other pm that we are knocking around.
        self.num_goals = max(num_objects, 1)
        goal_dim = 3 * self.num_goals

        high = np.ones([action_dim])
        self.action_space = spaces.Box(-high, high)
        high_obs = self.ENVIRONMENT_BOUNDS * np.ones([obs_dim])
        high_goal = self.ENVIRONMENT_BOUNDS * np.ones([goal_dim])

        env_range = [2.5, 2.5, 2]
        goal_range = [2, 2, 1.5]

        self.env_upper_bound = np.array(env_range)
        self.env_lower_bound = -self.env_upper_bound
        self.env_lower_bound[2] = 0  # set the y (updown) min to 0.
        self.goal_upper_bound = np.array(goal_range)
        self.goal_lower_bound = -self.goal_upper_bound
        self.goal_lower_bound[2] = 0  # set the y (updown) min to 0.
        self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([0.04])])
        self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([0.0])])


        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-high_goal, high_goal),
            achieved_goal=spaces.Box(-high_goal, high_goal),
            observation=spaces.Box(-high_obs, high_obs),
            controllable_achieved_goal=spaces.Box(-self.ENVIRONMENT_BOUNDS * np.ones([action_dim]),
                                                  self.ENVIRONMENT_BOUNDS * np.ones([action_dim])),
            full_positional_state=spaces.Box(-self.ENVIRONMENT_BOUNDS * np.ones([action_dim + 3 * self.num_objects]),
                                             self.ENVIRONMENT_BOUNDS * np.ones([action_dim + 3 * self.num_objects]))
        ))

        self.render_scene = False
        self.bullet_client = p
        self.physics_client_active = 0
        self.movable_goal = False
        self.roving_goal = False
        self.TARG_LIMIT = TARG_LIMIT
        self.TARG_MIN = 0.1
        self._seed()
        self.global_step = 0
        self.opposite_goal = False
        self.show_goal = True
        self.objects = []
        self.num_objects = num_objects
        self.state_representation = None
        self.sub_goals = None
        self.offset = np.array([0,0,0])
        if sparse:
            self.set_sparse_reward()


    def add_centering_offset(self, numbers):
        # numbers either come in as xyz, or xyzxyzxyz etc (in the case of goals or achieved goals)
        offset = np.array(list(self.offset) * (len(numbers) // 3))
        numbers = numbers + offset
        return numbers

    def subtract_centering_offset(self, numbers):
        offset = np.array(list(self.offset) * (len(numbers) // 3))
        numbers = numbers - offset
        return numbers

    def crop(self, num, lim):
        if num >= 0 and num < lim:
            num = lim
        elif num < 0 and num > -lim:
            num = -lim
        return num

    def reset_goal_pos(self, goal=None):

        # if self.state_representation:
        # 	random_ep = np.random.randint(0, len(self.episodes))
        # 	#random_frame = np.random.randint(0, len(self.episodes[random_ep]))
        # 	self.desired_frame = self.episodes[random_ep][-1][0]
        # 	self.desired_state = self.desired_frame['observation']
        # 	goal = self.desired_frame['achieved_goal']
        # 	print('set desired')
        if goal is None:
            self.goal = []
            for g in range(self.num_goals):
                goal = np.random.uniform(self.env_lower_bound, self.env_upper_bound)

                self.goal.append(goal)
            self.goal = np.concatenate(self.goal)

        else:
            self.goal = goal

        self.goal = np.array(self.goal)

        if self.render_scene:
            if self.show_goal:
                index = 0
                for g in range(0, self.num_goals):
                    self.bullet_client.resetBasePositionAndOrientation(self.goals[g],
                                                            self.goal[index:index+3], [0, 0, 0, 1])
                    self.bullet_client.changeConstraint(self.goal_cids[g], self.goal[index:index+3],
                                             maxForce=100)
                    index += 3

    def reset_object_pos(self, obs=None, extra_info=None, curric=False):

        if obs is None:
            index = 0
            for obj in self.objects:
                current_pos = self.bullet_client.getBasePositionAndOrientation(self.mass)[0]
                if curric == True:
                    vector_to_goal = np.array(
                        [self.goal[index] - current_pos[0], self.goal[index + 1] - current_pos[1], 0.6])

                    pos = np.array(current_pos) + vector_to_goal / 2 + (np.random.rand(3) * 1) - 0.5

                # shift it a little if too close to the goal
                pos = np.random.rand(3) * 4 - 2
                while self.calc_target_distance(pos[0:2], [self.goal[index], self.goal[index + 1]]) < 1:
                    pos = pos + (np.random.rand(3) * 1) - 0.5
                while self.calc_target_distance(pos[0:2], [current_pos[0], current_pos[1]]) < 1:
                    pos = pos + (np.random.rand(3) * 1) - 0.5
                pos[2] = 0.4
                ori = [0, 0, 0, 1]
                obs_vel_x, obs_vel_y = 0, 0
                self.bullet_client.resetBasePositionAndOrientation(obj, pos, ori)
                self.bullet_client.resetBaseVelocity(obj, [obs_vel_x, obs_vel_y, 0])
                index += 2
        else:
            starting_index = 4  # the first object index
            for obj in self.objects:
                obs_x, obs_y = obs[starting_index], obs[starting_index + 1]
                obs_z = 0.4 if extra_info is None else extra_info[0]
                pos = [obs_x, obs_y, obs_z]
                ori = [0, 0, 0, 1] if extra_info is None else extra_info[1:5]
                obs_vel_x, obs_vel_y = obs[starting_index + 2], obs[starting_index + 3]
                self.bullet_client.resetBasePositionAndOrientation(obj, pos, ori)
                self.bullet_client.resetBaseVelocity(obj, [obs_vel_x, obs_vel_y, 0])
                starting_index += 4  # go the the next object in the observation

    def initialize_actor_pos(self, o):
        x, y, x_vel, y_vel = o[0], o[1], o[2], o[3]
        self.bullet_client.resetBasePositionAndOrientation(self.mass, [x, y, -0.1], [0, 0, 0, 1])
        self.bullet_client.changeConstraint(self.mass_cid, [x, y, -0.1], maxForce=100)
        self.bullet_client.resetBaseVelocity(self.mass, [x_vel, y_vel, 0])

    # TODO change the env initialise start pos to a more general form of the function

    def initialize_start_pos(self, o, extra_info=None):
        if type(o) is dict:
            o = o['observation']
        self.initialize_actor_pos(o)
        if self.num_objects > 0:
            self.reset_object_pos(o, extra_info)

    def visualise_sub_goal(self, sub_goal, sub_goal_state='achieved_goal'):

        # in the sub_goal case we either only  have the positional info, or we have the full state positional info.
        # print(sub_goal)
        index = 0
        if self.sub_goals is None:
            self.sub_goals = []
            self.sub_goal_cids = []
            print('initing')
            sphereRadius = 0.15
            mass = 1
            colSphereId = self.bullet_client.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
            relativeChildPosition = [0, 0, 0]
            relativeChildOrientation = [0, 0, 0, 1]
            alpha = 0.5
            colors = [[212 / 250, 175 / 250, 55 / 250, alpha], [0, 1, 0, alpha], [0, 0, 1, alpha]]
            if sub_goal_state is 'achieved_goal':
                colors = [[0, 1, 0, alpha], [0, 0, 1, alpha]]

            for g in range(0, len(sub_goal) // 2):
                if g == 0 and sub_goal_state is not 'achieved_goal':  # all other ones include sphere
                    # the sphere
                    visId = p.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius,
                                                rgbaColor=colors[g])
                else:

                    visId = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.35],
                                                rgbaColor=colors[g])

                self.sub_goals.append(
                    self.bullet_client.createMultiBody(mass, colSphereId, visId, [sub_goal[index], sub_goal[index + 1], 0.1]))
                collisionFilterGroup = 0
                collisionFilterMask = 0
                self.bullet_client.setCollisionFilterGroupMask(self.sub_goals[g], -1, collisionFilterGroup, collisionFilterMask)
                self.sub_goal_cids.append(
                    self.bullet_client.createConstraint(self.sub_goals[g], -1, -1, -1, self.bullet_client.JOINT_FIXED,
                                             [sub_goal[index], sub_goal[index + 1], 0.1], [0, 0, 0.1],
                                             relativeChildPosition, relativeChildOrientation))
                index += 2


        else:

            for g in range(0, len(sub_goal) // 2):
                self.bullet_client.resetBasePositionAndOrientation(self.sub_goals[g], [sub_goal[index], sub_goal[index + 1], 0.1],
                                                        [0, 0, 0, 1])
                self.bullet_client.changeConstraint(self.sub_goal_cids[g], [sub_goal[index], sub_goal[index + 1], 0.1],
                                         maxForce=100)
                index += 2

    def calc_actor_state(self):
        # state = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)
        # pos, orn, pos_vel, orn_vel = state[0], state[1], state[-2], state[-1]
        # gripper_state = [self.bullet_client.getJointState(self.panda, 9)[0]]
        pos = self.bullet_client.getBasePositionAndOrientation(self.mass)[0]
        vel = self.bullet_client.getBaseVelocity(self.mass)[0]
        # return {'pos': self.subtract_centering_offset(pos), 'orn': orn, 'pos_vel': pos_vel, 'orn_vel': orn_vel,
        #         'gripper': gripper_state}
        return {'pos': self.subtract_centering_offset(pos), 'orn': 'nah', 'pos_vel': vel, 'orn_vel': 'nah',
                'gripper': np.array([0.04])}


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
            achieved_goal = np.concatenate([obj['pos'] for (i, obj) in env_state.items()])
            full_positional_state = np.concatenate([arm_state['pos'], achieved_goal])
        else:
            achieved_goal = arm_state['pos']
            full_positional_state = achieved_goal

        return_dict = {
            'observation': state.copy().astype('float32'),
            'achieved_goal': achieved_goal.copy().astype('float32'),
            'desired_goal': self.goal.copy().astype('float32'),
            'controllable_achieved_goal': arm_state['pos'].copy().astype('float32'),
            # just the x,y pos of the pointmass, the controllable aspects
            'full_positional_state': full_positional_state.copy().astype('float32')
        }

        if self.render_scene:
            time.sleep(0.01)  # TODO: get a camera angle.

        return return_dict

    def calc_target_distance(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance


    def compute_reward(self, achieved_goal, desired_goal, info=None):

        # reward given if new pos is closer than old

        current_distance = self.calc_target_distance(achieved_goal, desired_goal)

        position_reward = -1000 * (current_distance - self.last_target_distance)
        self.last_target_distance = current_distance

        # velocity_diff = self.calc_velocity_distance()
        # velocity_reward = -100*(velocity_diff - self.last_velocity_distance)
        # self.last_velocity_distance = velocity_diff
        # velocity_reward = self.calc_velocity_distance()

        # print('Vreward', velocity_reward)

        # if self.state_representation is not None:
        # 	position_reward = -tf.reduce_mean(tf.losses.MAE(self.state_representation(np.expand_dims(self.calc_state()['observation'],0))[0], self.state_representation(np.expand_dims(self.desired_state,0))[0]))

        return position_reward  # +velocity_reward

    def set_sparse_reward(self):
        print('Environment set to sparse reward')
        self.compute_reward = self.compute_reward_sparse

    def compute_reward_sparse(self, achieved_goal, desired_goal, info=None):

        # I know this should be vectorized but cbf atm - my version doesn't use vector ag and dg. Just need this for baselines
        # compatability.
        initially_vectorized = True
        if len(achieved_goal.shape) == 1:
            achieved_goal = np.expand_dims(np.array(achieved_goal), axis=0)
            desired_goal = np.expand_dims(np.array(desired_goal), axis=0)
            initially_vectorized = False

        reward = np.zeros(len(achieved_goal))

        for i in range(0, len(achieved_goal)):
            for g in range(0, len(achieved_goal[i]) // 3):
                g = g * 3  # increments of 2
                current_distance = self.calc_target_distance(achieved_goal[g:g + 3], desired_goal[g:g + 3])
                if current_distance > self.sparse_rew_thresh:
                    # TODO : DECIDE WHAT WE WANT HERE
                    # reward = -1
                    reward[i] -= 1

        if not initially_vectorized:
            return reward[0]
        else:
            return reward


    def goto(self, pos=None, orn=None, gripper=None):
        pos = list(pos)
        orn = list(orn)
        if pos and orn:
            pos = self.add_centering_offset(pos)
            jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll,
                                                                       ul,
                                                                       jr, rp, maxNumIterations=20)
            for i in range(pandaNumDofs):
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         jointPoses[i],
                                                         force=5 * 240.)
        if gripper:
            for i in [9, 10]:
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, gripper,
                                                         force=10)

    def step(self, action):


        # pos = action[0:3]
        # if self.use_orientation:
        #     orn = action[3:7]
        # else:
        #     orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
        # gripper = action[-1]
        # self.goto(pos, orn, gripper)
        action = action[0:3]
        current_pos = self.subtract_centering_offset(self.bullet_client.getBasePositionAndOrientation(self.mass)[0])

        new_pos = current_pos + action * 0.2
        new_pos = np.clip(self.add_centering_offset(new_pos), self.env_lower_bound, self.env_upper_bound)
        self.bullet_client.changeConstraint(self.mass_cid, new_pos, maxForce=100)


        for i in range(0, 20):
            self.bullet_client.stepSimulation()
        obs = self.calc_state()

        r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        current_distance = self.calc_target_distance(obs['achieved_goal'], self.goal)

        self.global_step += 1

        success = 0 if self.compute_reward_sparse(obs['achieved_goal'],
                                                  obs['desired_goal']) < 0 else 1  # assuming negative rewards

        # this part is only for interoperability with baselines
        if self.global_step == self._max_episode_steps:
            done = True
        else:
            done = False
        return obs, r, done, {'is_success': success}

    def reset(self):
        # self.bullet_client.resetSimulation()
        self.global_step = 0

        if self.physics_client_active == 0:

            if self.render_scene:
                self.bullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self.bullet_client = bullet_client.BulletClient(connection_mode=p.DIRECT)

            self.physics_client_active = 1

            sphereRadius = 0.2
            mass = 1
            visualShapeId = 2
            colSphereId = self.bullet_client.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
            self.mass = self.bullet_client.createMultiBody(mass, colSphereId, visualShapeId, [0, 0, 0.4])
            # objects = self.bullet_client.loadMJCF("/Users/francisdouglas/bullet3/data/mjcf/sphere.xml")
            # self.mass = objects[0]
            # self.mass = [p.loadURDF((os.path.join(urdfRoot,"sphere2.urdf")), 0,0.0,1.0,1.00000,0.707107,0.000000,0.707107)]
            relativeChildPosition = [0, 0, 0]
            relativeChildOrientation = [0, 0, 0, 1]
            self.mass_cid = self.bullet_client.createConstraint(self.mass, -1, -1, -1, self.bullet_client.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                     relativeChildPosition, relativeChildOrientation)

            alpha = 1
            colors = [[0, 1, 0, alpha], [0, 0, 1, alpha]]

            if self.show_goal:
                self.goals = []
                self.goal_cids = []

                for g in range(0, self.num_goals):
                    visId = p.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius,
                                                rgbaColor=colors[g])
                    self.goals.append(self.bullet_client.createMultiBody(mass, colSphereId, visId, [1, 1, 1.4]))
                    collisionFilterGroup = 0
                    collisionFilterMask = 0
                    self.bullet_client.setCollisionFilterGroupMask(self.goals[g], -1, collisionFilterGroup, collisionFilterMask)
                    self.goal_cids.append(
                        self.bullet_client.createConstraint(self.goals[g], -1, -1, -1, self.bullet_client.JOINT_FIXED, [1, 1, 1.4], [0, 0, 0],
                                                 relativeChildPosition, relativeChildOrientation))
            # self.bullet_client.setRealTimeSimulation(1)

            if self.num_objects > 0:

                colcubeId = self.bullet_client.createCollisionShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.35])
                for i in range(0, self.num_objects):
                    visId = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.35],
                                                rgbaColor=colors[i])
                    self.objects.append(self.bullet_client.createMultiBody(0.1, colcubeId, visId, [0, 0, 1.5]))

                # self.object = self.bullet_client.createMultiBody(mass,colSphereId,visualShapeId,[0.5,0.5,0.4])
                colwallId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 2.5, 0.5])
                wallvisId = 10
                wall = [p.createMultiBody(0, colwallId, 10, [self.TARG_LIMIT * 2 + 0.2, 0, 0.0],
                                          p.getQuaternionFromEuler([0, 0, 0]))]
                wall = [p.createMultiBody(0, colwallId, 10, [-self.TARG_LIMIT * 2 - 0.2, 0, 0.0],
                                          p.getQuaternionFromEuler([0, 0, 0]))]
                wall = [
                    p.createMultiBody(0, colwallId, 10, [0, self.TARG_LIMIT * 2 + 0.2, 0],
                                      p.getQuaternionFromEuler([0, 0, math.pi / 2]))]
                wall = [
                    p.createMultiBody(0, colwallId, 10, [0, -self.TARG_LIMIT * 2 - 0.2, 0],
                                      p.getQuaternionFromEuler([0, 0, math.pi / 2]))]

            if GUI:
                ACTION_LIMIT = 1
                self.x_shift = self.bullet_client.addUserDebugParameter("X", -ACTION_LIMIT, ACTION_LIMIT, 0.0)
                self.y_shift = self.bullet_client.addUserDebugParameter("Y", -ACTION_LIMIT, ACTION_LIMIT, 0.0)

            self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, GUI)

            self.bullet_client.setGravity(0, 0, -10)
            lookat = [0, 0, 0.1]
            distance = 7
            yaw = 0
            self.bullet_client.resetDebugVisualizerCamera(distance, yaw, -89, lookat)
            colcubeId = self.bullet_client.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 5, 0.1])
            visplaneId = self.bullet_client.createVisualShape(p.GEOM_BOX, halfExtents=[5, 5, 0.1], rgbaColor=[1, 1, 1, 1])
            plane = self.bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, 0, -0.2])

        # self.bullet_client.loadSDF(os.path.join(urdfRoot, "plane_stadium.sdf"))

        self.bullet_client.resetBasePositionAndOrientation(self.mass, [0, 0, 0.6], [0, 0, 0, 1])
        self.reset_goal_pos()

        # reset mass location
        if self.opposite_goal:
            x = -self.goal[0]
            y = -self.goal[1]
        else:
            x = self.crop(self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), self.TARG_MIN)
            y = self.crop(self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), self.TARG_MIN)
        x_vel = 0  # self.np_random.uniform(low=-1, high=1)
        y_vel = 0  # self.np_random.uniform(low=-1, high=1)

        self.initialize_actor_pos([x, y, x_vel, y_vel])
        if self.num_objects > 0:
            self.reset_object_pos()

        obs = self.calc_state()
        self.last_target_distance = self.calc_target_distance(obs['achieved_goal'], obs['desired_goal'])
        # self.last_velocity_distance = self.calc_velocity_distance()


        return obs


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

    def __init__(self, num_objects = 1, env_range = [0.2, 0.2, 0.2], goal_range = [0.2, 0.1, 0.2],
                 sparse=True, TARG_LIMIT=2, use_orientation=False, sparse_rew_thresh=0.02):
        fps = 240
        self.timeStep = 1. / fps
        self.render_scene = False
        self.physics_client_active = 0
        self.num_objects  = num_objects
        self.use_orientation = use_orientation
        self.sparse_reward_threshold = sparse_rew_thresh
        self.goal_range = goal_range
        self.env_range = env_range
        self.num_goals = max(self.num_objects, 1)
        self.env_upper_bound = np.array(env_range)
        self.env_lower_bound = -self.env_upper_bound
        self.env_lower_bound[1] = 0  # set the y (updown) min to 0.
        self.goal_upper_bound = np.array(goal_range)
        self.goal_lower_bound = -self.goal_upper_bound
        self.goal_lower_bound[1] = 0  # set the y (updown) min to 0.

        # Todo currently we do not use velocity. Lets see if that matters
        if use_orientation:
            self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1, 0.04])])
            self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1, 0.0])])
            arm_upper_obs_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1,1,1,1,1, 0.04])])  # includes velocity
            arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1,1,1,1,1, 0.0])])
            obj_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1])])
            obj_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1])])
        else:
            self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([0.04])])
            self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([0.0])])
            arm_upper_obs_lim = np.concatenate([self.env_upper_bound, np.array([1,1,1,0.04])]) # includes velocity
            arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 0.0])])
            obj_upper_lim = self.env_upper_bound
            obj_lower_lim = self.env_lower_bound

        upper_obs_dim = np.concatenate([arm_upper_obs_lim] + [obj_upper_lim] * self.num_objects)
        lower_obs_dim = np.concatenate([arm_lower_obs_lim] + [obj_lower_lim] * self.num_objects)
        upper_goal_dim = np.concatenate([self.env_upper_bound] * self.num_goals)
        lower_goal_dim = np.concatenate([self.env_lower_bound] * self.num_goals)

        self.action_space = spaces.Box(self.arm_lower_lim, self.arm_upper_lim)

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(lower_goal_dim, upper_goal_dim),
            achieved_goal=spaces.Box(lower_goal_dim, upper_goal_dim),
            observation=spaces.Box(lower_obs_dim, upper_obs_dim),
            controllable_achieved_goal=spaces.Box(self.arm_lower_lim, self.arm_upper_lim),
            full_positional_state=spaces.Box(lower_obs_dim, upper_obs_dim)
        ))

        if sparse:
            self.compute_reward = self.compute_reward_sparse


    def reset(self):

        if not self.physics_client_active:
            self.activate_physics_client()
            self.physics_client_active = True

        self.panda.reset()
        obs = self.panda.calc_state()

        return obs

    def render(self, mode):
        if (mode == "human"):
            self.render_scene = True
            return np.array([])
        if mode == 'rgb_array':
            raise NotImplementedError



    def step(self, action):
        self.panda.step(action)

        self.p.stepSimulation() # this is out here because the multiprocessing version will step everyone simulataneously.
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
            print('rendering m9----------------------')
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
            self.p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
            self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)


        self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
                                     cameraTargetPosition=[0.35, -0.13, 0])
        self.p.setAdditionalSearchPath(pd.getDataPath())

        self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, -9.8, 0)

        if self.num_objects == 0 :
            scene = default_scene
        elif self.num_objects == 1:
            scene = push_scene

        self.panda = pointMass3D(p, [0, 0, 0], scene,  self.arm_lower_lim, self.arm_upper_lim,
                                        self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
                                        self.goal_upper_bound,  self.use_orientation, self.render_scene)
        self.panda.control_dt = self.timeStep




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

        if not initially_vectorized:
            return reward[0]
        else:
            return reward

    def activate_human_interactive_mode(self):
        self.panda.step = self.panda.human_interactive_step


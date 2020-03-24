
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
from panda_sim import PandaSim
from pybullet_utils import bullet_client
import gym
from scenes import default_scene, push_scene, complex_scene
import gym.spaces as spaces

class pandaEnv(gym.GoalEnv):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 60
	}

	def __init__(self, num_objects = 1, env_range = [0.2, 0.2, 0.2], goal_range = [0.2, 0.1, 0.2], sparse=True, TARG_LIMIT=2, use_orientation=False, sparse_rew_thresh=0.02):
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
			obj_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1])])
			obj_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1])])
		else:
			self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([0.04])])
			self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([0.0])])
			obj_upper_lim = self.env_upper_bound
			obj_lower_lim = self.env_lower_bound

		upper_obs_dim = np.concatenate([self.arm_upper_lim] + [obj_upper_lim] * self.num_objects)
		lower_obs_dim = np.concatenate([self.arm_lower_lim] + [obj_lower_lim] * self.num_objects)
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

		self.panda.reset()

	def render(self):
		self.render_scene = True


	def step(self, action):
		self.panda.step(action)
		self.p.stepSimulation() # this is out here because the multiprocessing version will step everyone simulataneously.
		if self.render_scene:
			time.sleep(self.timeStep)

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

		self.panda = PandaSim(p, [0, 0, 0], scene,  self.arm_lower_lim, self.arm_upper_lim,
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

class pandaReach(pandaEnv):
	def __init__(self, num_objects = 0, env_range = [0.2, 0.2, 0.2], use_orientation=False):
		super().__init__(num_objects=num_objects, use_orientation=False)

class pandaPush(pandaEnv):
	def __init__(self, num_objects = 1, env_range = [0.2, 0.2, 0.2], goal_range = [0.2, 0.05, 0.2], use_orientation=False): # recall that y is up
		super().__init__(num_objects=num_objects, goal_range=goal_range, use_orientation=use_orientation)

panda = pandaPush()
panda.render()
panda.reset()
#panda.activate_human_interactive_mode()
for i in range (100000):
	action = panda.action_space.sample()
	obs, r, done, info = panda.step(action)
	print(obs, r)



	

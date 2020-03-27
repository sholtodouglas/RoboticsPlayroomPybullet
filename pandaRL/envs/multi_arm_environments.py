#
# import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# print("current_dir=" + currentdir)
# os.sys.path.insert(0, currentdir)
#
# import gym, gym.utils, gym.utils.seeding
# import pybullet as p
# import numpy as np
# import pybullet_data
# import time
# from pybullet_utils import bullet_client
# import pybullet_data as pd
# urdfRoot = pybullet_data.getDataPath()
# import gym.spaces as spaces
# import math
# from scenes import *
# from environments import *
#
#
# class pandaMultiEnv(gym.GoalEnv):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 60
#     }
#
#     def __init__(self, num_objects = 0, env_range_low = [-0.2, -0.1, -0.2], env_range_high = [0.2, 0.1, 0.2], goal_range_low = [-0.2, -0.1, -0.2], goal_range_high = [0.2, 0.1, 0.2],
#                  sparse=True, use_orientation=False, sparse_rew_thresh=0.05, pointMass = False, fixed_gripper = False):
#         fps = 240
#         self.timeStep = 1. / fps
#         self.render_scene = False
#         self.physics_client_active = 0
#         self.num_objects  = num_objects
#         self.use_orientation = use_orientation
#         self.fixed_gripper = fixed_gripper
#         self.sparse_reward_threshold = sparse_rew_thresh
#         self.num_goals = max(self.num_objects, 1)
#         self.num_rows = 1
#         self.num_arm_per_row = 1
#         obs_dim = 7
#         self.sparse_rew_thresh = sparse_rew_thresh
#         self._max_episode_steps = 250
#
#         obs_dim += 6 * num_objects  # pos and vel of the other pm that we are knocking around.
#         pos_step = 0.03
#         orn_step = 0.05
#         if self.use_orientation:
#             high = np.array([pos_step, pos_step, pos_step, orn_step,orn_step,orn_step,orn_step, 0.04])
#         else:
#             high = np.array([pos_step, pos_step, pos_step, 0.04])
#         self.action_space = spaces.Box(-high, high) # interesting how key this is. Oh because you know what, before it couldn't fucking go down. Duh.
#         self.pointMass = pointMass
#
#         self.env_upper_bound = np.array(env_range_high)
#         self.env_lower_bound = np.array(env_range_low)
#         #self.env_lower_bound[1] = 0  # set the y (updown) min to 0.
#         self.goal_upper_bound = np.array(goal_range_high)
#         self.goal_lower_bound = np.array(goal_range_low)
#         #self.goal_lower_bound[1] = 0  # set the y (updown) min to 0.
#
#
#         if use_orientation:
#             self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1, 0.04])])
#             self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1, 0.0])])
#             arm_upper_obs_lim = np.concatenate(
#                 [self.env_upper_bound, np.array([1, 1, 1, 1, 1, 1, 1, 0.04])])  # includes velocity
#             arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 1, 1, 1, 1, 0.0])])
#             obj_upper_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 1])])
#             obj_lower_lim = np.concatenate([self.env_lower_bound, -np.array([1, 1, 1, 1])])
#         else:
#             self.arm_upper_lim = np.concatenate([self.env_upper_bound, np.array([0.04])])
#             self.arm_lower_lim = np.concatenate([self.env_lower_bound, -np.array([0.0])])
#             arm_upper_obs_lim = np.concatenate([self.env_upper_bound, np.array([1, 1, 1, 0.04])])  # includes velocity
#             arm_lower_obs_lim = np.concatenate([self.env_upper_bound, -np.array([1, 1, 1, 0.0])])
#             obj_upper_lim = self.env_upper_bound
#             obj_lower_lim = self.env_lower_bound
#
#         upper_obs_dim = np.concatenate([arm_upper_obs_lim] + [obj_upper_lim] * self.num_objects)
#         lower_obs_dim = np.concatenate([arm_lower_obs_lim] + [obj_lower_lim] * self.num_objects)
#         upper_goal_dim = np.concatenate([self.env_upper_bound] * self.num_goals)
#         lower_goal_dim = np.concatenate([self.env_lower_bound] * self.num_goals)
#
#         lower_full_positional_state = np.concatenate([self.arm_lower_lim] + [obj_lower_lim] * self.num_objects) # like the obs dim, but without velocity.
#         upper_full_positional_state = np.concatenate([self.arm_upper_lim] + [obj_lower_lim] * self.num_objects)
#
#         #self.action_space = spaces.Box(self.arm_lower_lim, self.arm_upper_lim)
#
#         self.observation_space = spaces.Dict(dict(
#             desired_goal=spaces.Box(lower_goal_dim, upper_goal_dim),
#             achieved_goal=spaces.Box(lower_goal_dim, upper_goal_dim),
#             observation=spaces.Box(lower_obs_dim, upper_obs_dim),
#             controllable_achieved_goal=spaces.Box(self.arm_lower_lim, self.arm_upper_lim),
#             full_positional_state=spaces.Box( lower_full_positional_state, upper_full_positional_state)
#         ))
#
#
#         if sparse:
#             self.compute_reward = self.compute_reward_sparse
#
#
#
#     def reset(self):
#
#         if not self.physics_client_active:
#             self.activate_physics_client()
#             self.physics_client_active = True
#
#         [panda.reset() for panda in self.pandas]
#         obs = [panda.calc_state() for panda in self.pandas]
#
#         return obs
#
#     def render(self, mode):
#         if (mode == "human"):
#             self.render_scene = True
#             return np.array([])
#         if mode == 'rgb_array':
#             raise NotImplementedError
#
#
#
#     def step(self, action= None):
#
#         for i in range(len(self.pandas)):
#             self.pandas[i].step(action)
#
#         for i in range(0, 20):
#             self.p.stepSimulation()
#             # this is out here because the multiprocessing version will step everyone simulataneously.
#
#         if self.render_scene:
#             time.sleep(self.timeStep*3)
#
#         obs = [panda.calc_state() for panda in self.pandas]
#         for o in obs:
#             r = self.compute_reward(o['achieved_goal'], o['desired_goal'])
#             done = False
#             success = 0 if r < 0 else 1
#         # TODO: Should it be done if we exceed the environment limits to prevent bad data?
#         return obs, r, done, {'is_success': success}
#
#     def activate_physics_client(self):
#
#         if self.render_scene:
#
#             self.p = bullet_client.BulletClient(connection_mode=p.GUI)
#             self.p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
#             self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#         else:
#             self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
#
#
#         #self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
#         # self.p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
#         #                              cameraTargetPosition=[0.35, -0.13, 0])
#         self.p.setAdditionalSearchPath(pd.getDataPath())
#
#         self.p.setTimeStep(self.timeStep)
#         self.p.setGravity(0, -9.8, 0)
#
#         if self.num_objects == 0 :
#             scene = default_scene
#         elif self.num_objects == 1:
#             scene = push_scene
#         elif self.num_objects == 2:
#             scene = complex_scene
#
#         X = 0
#         Y = 0
#         Z = 0
#         spacing = 1
#         self.pandas = []
#         for i in range(0,self.num_rows):
#             for j in range(self.num_arm_per_row):
#                 self.pandas.append(pointMassSim(self.p, [X,Y,Z], scene,  self.arm_lower_lim, self.arm_upper_lim,
#                                         self.env_lower_bound, self.env_upper_bound, self.goal_lower_bound,
#                                         self.goal_upper_bound,  self.use_orientation, self.render_scene, pointMass = self.pointMass, fixed_gripper=self.fixed_gripper))
#                 X = X + spacing
#             X = 0
#             Z = Z - spacing
#
#         for panda in self.pandas:
#             panda.control_dt = self.timeStep
#         lookat = [spacing*2/3, 0, -spacing*2/3]
#         distance = 3
#         yaw = 140
#         self.p.resetDebugVisualizerCamera(distance, yaw, -150, lookat)
#
#
#
#
#     def calc_target_distance(self, achieved_goal, desired_goal):
#         distance = np.linalg.norm(achieved_goal - desired_goal)
#         return distance
#
#     def compute_reward(self, achieved_goal, desired_goal):
#         return -self.calc_target_distance(achieved_goal,desired_goal)
#
#
#     def compute_reward_sparse(self, achieved_goal, desired_goal, info=None):
#         # I know this should be vectorized but cbf atm - my HER doesn't use vector ag and dg. Just need this for baselines
#         # compatability.
#         # This implements a piecewise reward function, for multiobject envs.
#         initially_vectorized = True
#         dimensions = 3
#         if len(achieved_goal.shape) == 1:
#             achieved_goal = np.expand_dims(np.array(achieved_goal), axis=0)
#             desired_goal = np.expand_dims(np.array(desired_goal), axis=0)
#             initially_vectorized = False
#         reward = np.zeros(len(achieved_goal))
#         for i in range(0, len(achieved_goal)):
#             for g in range(0, len(achieved_goal[i]) // dimensions ): # three for three dimensions
#                 g = g * dimensions   # increments of 2
#                 current_distance = self.calc_target_distance(achieved_goal[g:g + dimensions ], desired_goal[g:g + dimensions ])
#                 if current_distance > self.sparse_reward_threshold:
#                     reward[i] -= 1
#                 if current_distance < self.sparse_reward_threshold:
#                     reward[i] -= current_distance # semi sparse reward, rewards centering it exactly!
#
#         if not initially_vectorized:
#             return reward[0]
#         else:
#             return reward
#
#     def activate_human_interactive_mode(self):
#         self.panda.step = self.panda.human_interactive_step
#
#     def visualise_sub_goal(self, sub_goal, sub_goal_state = 'full_positional_state'):
#         self.panda.visualise_sub_goal(sub_goal, sub_goal_state = sub_goal_state)
#
# def main():
#     panda = pandaMultiEnv()
#     controls = []
#     render = True
#     if render:
#         panda.render(mode='human')
#     panda.reset()
#     if render:
#         controls.append(panda.p.addUserDebugParameter("X", panda.action_space.low[0], panda.action_space.high[0], 0))
#         controls.append(panda.p.addUserDebugParameter("Y", panda.action_space.low[1], panda.action_space.high[1], 0.02))
#         controls.append(panda.p.addUserDebugParameter("Z", panda.action_space.low[2], panda.action_space.high[2], 0))
#         controls.append(panda.p.addUserDebugParameter("grip", panda.action_space.low[3], panda.action_space.high[3], 0))
#     state_control = False #True
#
#     if state_control:
#         panda.activate_human_interactive_mode()
#
#     for i in range(100000):
#         panda.reset()
#         #panda.visualise_sub_goal(np.array([0,0,0,0.04]), 'controllable_achieved_goal')
#         t1 = time.time()
#         for j in range(0, 150):
#
#             action = []
#             if state_control:
#                 panda.step()
#                 time.sleep(0.005)
#             else:
#                 if render:
#                     for i in range(0, len(controls)):
#                         action.append(panda.p.readUserDebugParameter(i))
#                 action = panda.action_space.sample()
#                 obs, r, done, info = panda.step(np.array(action))
#                 #print(obs['achieved_goal'], obs['desired_goal'], r)
#                 #time.sleep(0.01)
#         print(time.time()-t1)
#
#
# if __name__ == "__main__":
#     main()
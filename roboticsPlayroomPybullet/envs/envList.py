import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

from environments import playEnv

class pandaReach(playEnv):
	def __init__(self, num_objects = 0):
		super().__init__(num_objects=num_objects, use_orientation=False)

class pandaPush(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-0.18, -0.18, -0.055], env_range_high = [0.18, 0.18, -0.04],  goal_range_low=[-0.1, -0.1, -0.06], goal_range_high = [0.1, 0.1, -0.05], use_orientation=False): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = goal_range_low, obj_upper_bound = goal_range_high) # in push restrict the inti positions slightly.

class pandaPick(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-0.18, -0.18, -0.055], env_range_high = [0.18, 0.18, 0.2],  goal_range_low=[-0.18, -0.18, 0.0], goal_range_high = [0.18, 0.18, 0.1], use_orientation=False): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = goal_range_low, obj_upper_bound = goal_range_high)

class pandaReach2D(playEnv):
	def __init__(self, num_objects = 0, env_range_low = [-0.18, -0.18, -0.07], env_range_high = [0.18, 0.18, 0.0],  goal_range_low=[-0.18, -0.18, -0.06], goal_range_high = [0.18, 0.18, -0.05], use_orientation=False): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high, goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation)

class pandaPlay(playEnv):
	def __init__(self, num_objects = 2, env_range_low = [-1.0, -1.0, -0.4], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='absolute_quat', show_goal=False)


class pandaPlayRelJoints(playEnv):
	def __init__(self, num_objects = 2, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='relative_joints', show_goal=False)

class pandaPlayRelJoints1Obj(playEnv):
    def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
        super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                        goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                        obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='relative_joints', show_goal=False)

class pandaPlayAbsJoints1Obj(playEnv):
    def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
        super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                        goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                        obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='absolute_joints', show_goal=False)


class pandaPlay1Obj(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='absolute_quat', show_goal=False)

class pandaPlayRel1Obj(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False, max_episode_steps=None, play=True, action_type='relative_quat', show_goal=False)

class pandaPlayAbsRPY1Obj(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_rpy', show_goal=False)

class pandaPlayRelRPY1Obj(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                          max_episode_steps=None, play=True, action_type='relative_rpy', show_goal=False)


class UR5Reach(playEnv):
	def __init__(self, num_objects = 0):
		super().__init__(num_objects=num_objects, use_orientation=False, arm_type='UR5')

class UR5PlayAbsRPY1Obj(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_rpy', show_goal=False, arm_type='UR5')

class UR5PlayRelRPY1Obj(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                          max_episode_steps=None, play=True, action_type='relative_rpy', show_goal=False, arm_type='UR5')

class UR5PlayRelJoints1Obj(playEnv):
    def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
        super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                        goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                        obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='relative_joints', show_goal=False, arm_type='UR5')

class UR5PlayAbsJoints1Obj(playEnv):
    def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
        super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                        goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                        obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_joints', show_goal=False, arm_type='UR5')


class UR5Play1Obj(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='absolute_quat', show_goal=False, arm_type='UR5')

class UR5PlayRel1Obj(playEnv):
	def __init__(self, num_objects = 1, env_range_low = [-1.0, -1.0, -0.2], env_range_high = [1.0, 1.0, 1.0],
                 goal_range_low= [-0.18, 0, 0.05], goal_range_high = [0.18, 0.3, 0.1], use_orientation=True): # recall that y is up
		super().__init__(num_objects=num_objects, env_range_low = env_range_low, env_range_high = env_range_high,
                         goal_range_low=goal_range_low, goal_range_high=goal_range_high, use_orientation=use_orientation,
                         obj_lower_bound = [-0.18, 0, 0.05], obj_upper_bound = [0.18, 0.3, 0.1], return_velocity=False,
                         max_episode_steps=None, play=True, action_type='relative_quat', show_goal=False, arm_type='UR5')
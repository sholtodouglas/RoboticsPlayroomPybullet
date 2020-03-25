import time
import numpy as np
import math


useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

# Todo the different magnitudes of the action dimensions will impact BC loss

class PandaSim(object):
  def __init__(self, bullet_client, offset, load_scene,  arm_lower_lim, arm_upper_lim, env_lower_bound,
               env_upper_bound, goal_lower_bound, goal_upper_bound, use_orientation = False,
               render_scene = False):


    self.bullet_client = bullet_client
    self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
    self.offset = np.array(offset)
    #print("offset=",offset)
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    self.objects = load_scene(self.bullet_client, offset,flags)
    self.num_objects = len(self.objects)
    self.num_goals = max(self.num_objects, 1)

    self.arm_lower_lim = arm_lower_lim
    self.arm_upper_lim = arm_upper_lim
    self.env_upper_bound = env_upper_bound
    self.env_lower_bound = env_lower_bound
    self.goal_upper_bound = goal_upper_bound
    self.goal_lower_bound = goal_lower_bound

    self.render_scene = render_scene
    self.use_orientation = use_orientation



    orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
    self.endEffectorIndex = 11
    self.state = 0
    self.control_dt = 1./240.
    self.finger_target = 0
    self.gripper_height = 0.2
    # in addition to the offset of the actual environment, if we want to keep our obs and actions zero centered
    # then we need a readout offset. x and y are correct (when the offset is taken into account), but z is centered
    # around -0.6 from the offset. ANY TIME WE GIVE OR RECEIVE xyz position, we must take this into account.
    # when we get a measurement, subtract, when we give a location, add.
    self.offset = self.offset + np.array([0,0.0,-0.6]) # this is the centering offset
    #create a constraint to keep the fingers centered
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

    # create the goal objects
    self.show_goal = True
    alpha = 1
    self.obj_colors = [[0, 1, 0, alpha], [0, 0, 1, alpha]] # colors for two objects
    if self.render_scene and self.show_goal:
      sphereRadius = 0.02
      mass = 1
      colSphereId = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_SPHERE, radius=sphereRadius)
      relativeChildOrientation = [0, 0, 0, 1]
      self.goals = []
      self.goal_cids = []
      collisionFilterGroup = 0
      collisionFilterMask = 0
      for g in range(self.num_goals):
        init_loc = self.add_centering_offset(np.array([0,0.2,0]))
        visId = self.bullet_client.createVisualShape(self.bullet_client.GEOM_SPHERE, radius = sphereRadius, rgbaColor=self.obj_colors[g])
        self.goals.append(self.bullet_client.createMultiBody(mass, colSphereId, visId, init_loc))

        self.bullet_client.setCollisionFilterGroupMask(self.goals[g], -1, collisionFilterGroup, collisionFilterMask)
        self.goal_cids.append(
        self.bullet_client.createConstraint(self.goals[g], -1, -1, -1, self.bullet_client.JOINT_FIXED,  [0, 0, 0],[0,0,0],
                                 init_loc, relativeChildOrientation))
    self.reset()


  def add_centering_offset(self, numbers):
    # numbers either come in as xyz, or xyzxyzxyz etc (in the case of goals or achieved goals)
    offset = np.array(list(self.offset)*(len(numbers)//3))
    numbers = numbers + offset
    return numbers

  def subtract_centering_offset(self, numbers):
    offset = np.array(list(self.offset) * (len(numbers) // 3))
    numbers = numbers - offset
    return numbers

  def reset_goal_positions(self, goal=None):
    if goal is None:
      self.goal = []
      for g in range(self.num_goals):

        goal = np.random.uniform(self.env_lower_bound, self.env_upper_bound)
        self.goal.append(goal)
      self.goal = np.concatenate(self.goal)
    else:

      self.goal = np.array(self.add_centering_offset(goal))


    if self.render_scene and self.show_goal:
      index = 0
      for g in range(self.num_goals):
        pos = self.add_centering_offset(self.goal[index:index+3])
        self.bullet_client.resetBasePositionAndOrientation(self.goals[g], pos, [0,0,0,1])
        self.bullet_client.changeConstraint(self.goal_cids[g], pos, maxForce = 100 )
        index += 3


  def reset_arm(self):
    index = 0
    for j in range(self.bullet_client.getNumJoints(self.panda)):
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
      info = self.bullet_client.getJointInfo(self.panda, j)
      # print("info=",info)
      jointName = info[1]
      jointType = info[2]
      if (jointType == self.bullet_client.JOINT_PRISMATIC):
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
        index = index + 1
      if (jointType == self.bullet_client.JOINT_REVOLUTE):
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
        index = index + 1


  def reset_object_positions(self, obs=None):
    # Todo object velocities to make this properly deterministic
    if obs is None:
      height_offset = 0.03
      for o in self.objects:
        pos = self.add_centering_offset(np.random.uniform(self.env_lower_bound, self.env_upper_bound))
        pos[1] = pos[1] + height_offset # so they don't collide
        self.bullet_client.resetBasePositionAndOrientation(o, pos, [0,0,0,1])
        height_offset += 0.03
      for i in range(0,10):
        self.bullet_client.stepSimulation()
    else:
      if self.use_orientation:
        index = 8
        increment = 7
      else:
        index = 4
        increment = 3
      for o in self.objects:
        pos = obs[index:index+3]
        if self.use_orientation:
          orn = obs[index+3:index+7]
        else:
          orn = [0,0,0,1]
        self.bullet_client.resetBasePositionAndOrientation(o, pos, orn)
        index += increment


  def reset(self):
    self.reset_arm()
    self.reset_goal_positions()
    self.reset_object_positions()
    self.t = 0.

  def calc_arm_state(self):
    state = self.bullet_client.getLinkState(self.panda, self.endEffectorIndex, computeLinkVelocity=1)
    pos, orn, pos_vel, orn_vel = state[0], state[1], state[-2], state[-1]
    gripper_state = [self.bullet_client.getJointState(self.panda, 9)[0]]

    return {'pos':self.subtract_centering_offset(pos), 'orn':orn, 'pos_vel': pos_vel, 'orn_vel':orn_vel, 'gripper': gripper_state}

  def calc_environment_state(self):
    object_states = {}
    for i in range(0, self.num_objects):
      pos, orn  = self.bullet_client.getBasePositionAndOrientation(self.objects[i])
      object_states[i] = {'pos':self.subtract_centering_offset(pos), 'orn':orn}

    return object_states


  def calc_state(self):
    arm_state = self.calc_arm_state()
    if self.use_orientation:
      arm_elements = ['pos', 'orn', 'gripper']
    else:
      arm_elements = ['pos', 'gripper']
    state = np.concatenate([np.array(arm_state[i]) for i in arm_elements])
    if self.num_objects > 0:
      env_state = self.calc_environment_state()

      if self.use_orientation:
        obj_states = np.concatenate([np.concatenate([obj['pos'], obj['orn']]) for (i,obj) in env_state.items()])
      else:
        obj_states = np.concatenate([obj['pos'] for (i,obj) in env_state.items()])
      state = np.concatenate([state,obj_states])
      achieved_goal = np.concatenate([obj['pos'] for (i,obj) in env_state.items()])
      full_positional_state = np.concatenate([arm_state['pos'], achieved_goal])
    else:
      achieved_goal = arm_state['pos']
      full_positional_state = achieved_goal

    return_dict= {
	            'observation': state.copy().astype('float32'),
	            'achieved_goal': achieved_goal.copy().astype('float32'),
	            'desired_goal':  self.goal.copy().astype('float32'),
				'controllable_achieved_goal': arm_state['pos'].copy().astype('float32'), # just the x,y pos of the pointmass, the controllable aspects
				'full_positional_state': full_positional_state.copy().astype('float32')
            }

    if self.render_scene:
      pass #TODO: get a camera angle.

    return return_dict

  def update_state(self):
    keys = self.bullet_client.getKeyboardEvents()
    print(keys)
    if len(keys)>0:
      for k,v in keys.items():
        if v&self.bullet_client.KEY_WAS_TRIGGERED:
          if (k==ord('1')):
            self.state = 1
          if (k==ord('2')):
            self.state = 2
          if (k==ord('3')):
            self.state = 3
          if (k==ord('4')):
            self.state = 4
          if (k==ord('5')):
            self.state = 5
          if (k==ord('6')):
            self.state = 6
          if (k==ord('8')):
            self.state = 8



        if v&self.bullet_client.KEY_WAS_RELEASED:
            self.state = 0

  def goto(self, pos=None, orn=None, gripper=None):
    pos= list(pos)
    orn = list(orn)
    if pos and orn:
      pos = self.add_centering_offset(pos)
      jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll, ul,
                                                                 jr, rp, maxNumIterations=20)
      for i in range(pandaNumDofs):
        self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],
                                                 force=5 * 240.)
    if gripper:
      for i in [9,10]:
        self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,gripper ,force= 10)

  def step(self, action):
    action = np.clip(action, self.arm_lower_lim, self.arm_upper_lim) # clip because most RL implementations only scale by the high extent of the dim.
    pos = action[0:3]
    if self.use_orientation:
      orn = action[3:7]
    else:
      orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
    gripper = action[-1]
    self.goto(pos,orn,gripper)

  def grasp(self, object):
    raise NotImplementedError
  def human_interactive_step(self):

    if self.state==6:
      self.finger_target = 0.01
    if self.state==5:
      self.finger_target = 0.04

    self.update_state()
    if self.state == 8:
      self.reset()
      return
    print("self.state=",self.state)
    print("self.finger_target=",self.finger_target)
    alpha = 0.9 #0.99
    pos, orn = None, None
    if self.state==1 or self.state==2 or self.state==3 or self.state==4 or self.state==7:
      #gripper_height = 0.034
      self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.03
      if self.state == 2 or self.state == 3 or self.state == 7:
        self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.2
      
      t = self.t
      self.t += self.control_dt

      pos = [0.2 * math.sin(1.5 * t), self.gripper_height-self.offset[1],  0.1 * math.cos(1.5 * t)]

      if self.state == 3 or self.state== 4:
        pos, o = self.bullet_client.getBasePositionAndOrientation(self.objects[0])
        pos = self.subtract_centering_offset(pos)
        pos = [pos[0], self.gripper_height-self.offset[1], pos[2]]

        self.prev_pos = pos
      if self.state == 7:
        pos = self.prev_pos
        diffX = pos[0] #- self.offset[0]
        diffZ = pos[2] #- (self.offset[2]-0.6)
        self.prev_pos = [self.prev_pos[0] - diffX*0.1, self.prev_pos[1], self.prev_pos[2]-diffZ*0.1]

      	
      orn = self.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
    self.goto(pos, orn, self.finger_target)

      #target for fingers




class PandaSimAuto(PandaSim):
  def __init__(self, bullet_client, offset):
    PandaSim.__init__(self, bullet_client, offset)
    self.state_t = 0
    self.cur_state = 0
    self.states=[0,3,5,4,6,3,7]
    self.state_durations=[1,1,1,2,1,1, 10]
  
  def update_state(self):
    self.state_t += self.control_dt
    if self.state_t > self.state_durations[self.cur_state]:
      self.cur_state += 1
      if self.cur_state>=len(self.states):
        self.cur_state = 0
      self.state_t = 0
      self.state=self.states[self.cur_state]
      print("self.state=",self.state)


class pointMass3D(PandaSim):
  def __init__(self, bullet_client, offset, load_scene,  arm_lower_lim, arm_upper_lim, env_lower_bound, env_upper_bound, goal_lower_bound, goal_upper_bound, use_orientation = False, render_scene = False):
    PandaSim.__init__(self, bullet_client, offset, load_scene,  arm_lower_lim, arm_upper_lim, env_lower_bound, env_upper_bound, goal_lower_bound, goal_upper_bound, use_orientation = use_orientation, render_scene = render_scene)


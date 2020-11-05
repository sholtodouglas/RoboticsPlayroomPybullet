import pandaRL
import gym
import numpy as np
import time


env = gym.make('UR5PlayAbsRPY1Obj-v0')
env.render('human')
env.reset()

examples = np.load('collected_data/pick.npz')

obs = examples['obs']
acts = examples['acts']
goals = examples['desired_goals']
ags = examples['achieved_goals']
full_positional_states = examples['full_positional_states']

for i in range(0,100):
    idx = np.random.choice(len(obs))
    traj = obs[idx,:,:]
    env.reset(traj[0])


    for i in range(len(traj)):
        env.reset(traj[i])
        # env.visualise_sub_goal(full_positional_states[idx, -1,:])
        env.panda.reset_goal_pos(ags[idx, -1, :])
        #env.step(acts[idx, i, :])
        time.sleep(0.02)
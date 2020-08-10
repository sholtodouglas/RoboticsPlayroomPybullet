# Ok, lets collect data from preprogrammed pick

import gym
import pandaRL
import numpy as np
import os
import shutil
import os 

try:
     os.makedirs('collected_data/pick_demos')
except:
    pass
env = gym.make('pandaPick-v0')
env.render(mode='human')
env.reset()

open_gripper = np.array([0.04])
closed_gripper = np.array([0.01])

def go_above(env, obj_number):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos'] + np.array([0, 0.05, 0])
    current_position = env.panda.calc_actor_state()['pos']
    current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position - current_position, np.array(env.panda.default_arm_orn)-np.array(current_orn), open_gripper])
    return action

def descend(env, obj_number):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos'] + np.array([0, -0.015, 0])
    current_position = env.panda.calc_actor_state()['pos']
    current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position - current_position, np.array(env.panda.default_arm_orn)-np.array(current_orn), open_gripper])
    return action

def close(env, obj_number):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos']
    current_position = env.panda.calc_actor_state()['pos']
    current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position - current_position, np.array(env.panda.default_arm_orn)-np.array(current_orn), closed_gripper])
    return action

def lift(env, obj_number):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos']
    desired_position[1] +=  0.02
    current_position = env.panda.calc_actor_state()['pos']
    current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position - current_position, np.array(env.panda.default_arm_orn)-np.array(current_orn), closed_gripper])
    return action

def take_to(env, position):
    desired_position = position
    current_position = env.panda.calc_actor_state()['pos']
    current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position - current_position, np.array(env.panda.default_arm_orn)-np.array(current_orn), closed_gripper])*0.5
    return action


times = [1,2,3,4,6]
states  = [go_above, descend, close, lift, take_to]
p = env.panda.bullet_client
t = 0
state_pointer = 0
dt = 2/10

action_buff = []
observation_buff = []
desired_goals_buff = []
achieved_goals_buff = []
controllable_achieved_goal_buff = []
full_positional_state_buff = []

base_path = 'collected_data/pick_demos/'
demo_count = len(list(os.listdir(base_path)))


for i in range(0,500):
    o = env.reset()
    
    take_to_pos =  np.random.uniform(env.goal_lower_bound, env.goal_upper_bound)
    
    env.panda.reset_goal_pos(take_to_pos)
    t = 0
    state_pointer = 0

    acts, obs, goals, ags , cagb,  fpsb = [], [], [], [], [], []
    example_path = base_path+str(demo_count)
    os.makedirs(example_path)
    os.makedirs(example_path+'/env_states')
    counter = 0
    while(t < times[state_pointer]):
        if state_pointer == 4:
            action = states[state_pointer](env,take_to_pos)
        else:
            action = states[state_pointer](env, obj_number = 0)

        p.saveBullet(example_path+'/env_states/'+str(counter)+".bullet")
        counter += 1 # little counter for saving the bullet states
        o2, r, d, _ = env.step(action)
        if d:
            print('Demo failed')
            # delete the folder with all the saved states within it
            shutil.rmtree(base_path + str(demo_count))
            break
        acts.append(action), obs.append(o['observation']), goals.append(o['desired_goal']), ags.append(o2['achieved_goal']) ,\
        cagb.append(o2['controllable_achieved_goal']), fpsb.append(o2['full_positional_state'])
        o = o2

        t += dt
        if t >= times[state_pointer]:
            state_pointer += 1

        if t >=  times[-1]:
            if r > -1: # if demo was successful, append
                print('Storing Replay')
                action_buff.append(acts), observation_buff.append(obs), desired_goals_buff.append(goals), achieved_goals_buff.append(ags), \
                controllable_achieved_goal_buff.append(cagb), full_positional_state_buff.append(fpsb)

                np.savez(base_path+str(demo_count) +'/data', acts=acts, obs=obs,
                         desired_goals=goals,
                         achieved_goals=ags,
                         controllable_achieved_goals=cagb,
                         full_positional_states=fpsb)
                demo_count += 1
            else:
                print('Demo failed')
                # delete the folder with all the saved states within it
                shutil.rmtree(base_path + str(demo_count))

            break




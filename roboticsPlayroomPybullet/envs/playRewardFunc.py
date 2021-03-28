import numpy as np
import pybullet as p

'''
A set of functions which test various elements of the pybullet state space for success
'''

# Indices in 'achieved goal'
block = [0,3]
qqqq = [3,7]
drawer = 7
door = 8
button = 9
dial = 10

def compare_xyz(g, ag, limits = np.array([0.05, 0.05, 0.05])):
    if (abs(g-ag) > limits).any():
        #print('Failed xyz')
        return False
    else:
        return True
    
    
def compare_RPY(g, ag, limits = np.array([np.pi/4,np.pi/4,np.pi/4])):
    g = np.array(p.getEulerFromQuaternion(g))
    ag = np.array(p.getEulerFromQuaternion(ag))
    if (abs(g-ag) > limits).any():
        #print('Failed rpy')
        return False
    else:
        return True
    
def compare_drawer(g, ag, limit=0.025):
    if abs(g-ag) > limit:
        #print('Failed drawer')
        return False
    else:
        return True
    
def compare_door(g, ag, limit=0.03):
    if abs(g-ag) > 0.04:
        #print('Failed door', g, ag)
        return False
    else:
        return True
    
    
def compare_button(g, ag, limit=0.01):
    if abs(g-ag) >limit: 
        #print('Failed button', g , ag)
        return False
    else:
        return True
    
def compare_dial(g,ag, limit=0.3):
    if abs(g-ag) > limit:
        #print('Failed dial')
        return False
    else:
        return True
    
    
'''
Could easily convert this to piece wise function if desired - currently sparse for complete success
'''
def success_func(ag, g):
    g,ag = np.squeeze(g), np.squeeze(ag)
    if compare_xyz(g[block[0]:block[1]], ag[block[0]:block[1]])\
     and compare_RPY(g[qqqq[0]:qqqq[1]], ag[qqqq[0]:qqqq[1]]) \
     and compare_drawer(g[drawer], ag[drawer]) \
     and compare_door(g[door], ag[door]) \
     and compare_button(g[button], ag[button]) \
     and compare_dial(g[dial], ag[dial]):
     
        return 0
    else:
        return -1
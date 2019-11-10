# =================================================================================
''' Mountain Car environment related functions
'''
# =================================================================================
import os, math, gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.classic_control.mountain_car import MountainCarEnv
# =================================================================================
#   Written by Ran Wei at rw422@tamu.edu, 6/13/2019
# =================================================================================
class MountainCar_v1(MountainCarEnv):
    """ Mountain Car environment with modified reset function.
    """
    global initial_state, initial_gravity
    initial_state = -0.5
    initial_gravity = 0.0025
    
    def __init__(self):
        super().__init__()  
        self.goal_position = 0.5 
    
    def reset(self,coef_state = 0, coef_gravity = 1, *arg):
        """ coef_state: adjustment added to initial_state, initial_state = -0.5 + coef_state
            coef_gravity: gravity multiplier, gravity = coef_gravity * 0.0025
        """
        super().reset()
        self.state = np.array([initial_state + coef_state,0]) # inital state variable
        self.gravity = initial_gravity * coef_gravity # make the car 2 times heavier
        return np.array(self.state)


# =================================================================================
# from environment
# =================================================================================
def obs2state(observation):
    """ 
    Map observation to state index in 19(position)*15(velocity) state space grid. 
    Range = [0, 284].
    Input: [position, speed].
    """
    # Define state space
    num_states = [19, 15]
    min_observation = [-1.2, -0.07]

    state = (observation - np.array(min_observation))*np.array([10, 100])
    state = np.round(state, 0).astype(int)
    return state[0]*num_states[1] + state[1]


def state2obs(idx_state): 
    ''' map state index in 19(position)*15(velocity) state space grid to rounded observations
    ''' 
    min_observation = [-1.2, -0.07]
    max_observation = [0.6, 0.07]
    position_space = np.round(np.arange(min_observation[0], max_observation[0]+0.05, 0.1),1)
    velocity_space = np.round(np.arange(min_observation[1], max_observation[1], 0.01), 2)
    
    position = int(np.floor(idx_state/velocity_space.shape[0]))
    position = position_space[position]
    velocity = idx_state%velocity_space.shape[0]
    velocity = velocity_space[velocity]
    return [position, velocity]


def sample_transition_matrix(coef_gravity = 1, save = False):
    ''' sample state 0 from normal distribution, calculate transition probability using fraction
        add 0.0001 to diagnal where value is 0
    '''
    env = MountainCar_v1()
    gravity = 0.0025 * coef_gravity
    force = 0.001

    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    
    # ============= env dynamics ==============================================
    def get_velocity(position, velocity, action):
        velocity += (action-1)*force + math.cos(3*position)*(-gravity)
        velocity = np.clip(velocity, -0.07, 0.07)
        return velocity

    def get_position(position, new_velocity):
        position += new_velocity
        position = np.clip(position, -1.2, 0.6)
        return position
    # ==========================================================================
    position_space = np.round(np.arange(min_position, max_position+0.05, 0.1),1)
    velocity_space = np.round(np.arange(-max_speed, max_speed, 0.01),2)

    transition_mat = np.zeros((285, 285, 3))
    for position in position_space:
        if position == max_position: 
            sample_range = [position - 0.0499999, position]
        elif position == min_position: 
            sample_range = [position, position + 0.0499999]
        else: 
            sample_range = [position - 0.0499999, position + 0.0499999]
        position_sample = np.random.uniform(sample_range[0], sample_range[1], 100)
        
        for velocity in velocity_space:
            if velocity == max_speed: 
                sample_range = [velocity - 0.00499999, velocity]
            elif velocity == -max_speed:
                sample_range = [velocity, velocity + 0.00499999]
            else: 
                sample_range = [velocity - 0.00499999, velocity + 0.00499999]
            velocity_sample = np.random.uniform(sample_range[0], sample_range[1], 100)
            
            print('calulating position {}, velocity {}'.format(position, velocity))
            for action in [0,1,2]:
                old_states, new_states = [], []
                for ran_position in position_sample: 
                    for ran_velocity in velocity_sample: 
                        next_velocity = get_velocity(ran_position, ran_velocity, action)
                        next_position = get_position(ran_position, next_velocity)
                        idx_old_state = obs2state([ran_position, ran_velocity])
                        idx_new_state = obs2state([next_position, next_velocity])
                        old_states.append(idx_old_state)
                        new_states.append(idx_new_state)
                unique_old_state = np.unique(old_states)
                if len(unique_old_state) > 1:
                    print(unique_old_state)
                unique_new_states = np.unique(new_states)
                for unique_new_state in unique_new_states: 
                    prob_new_state = np.sum(np.array(new_states) == unique_new_state)/len(new_states)
                    transition_mat[unique_old_state, unique_new_state, action] = prob_new_state
    
    # handeling zero self transition
    # for action in [0,1,2]: 
    #     for i in range(285): 
    #         if transition_mat[i,i,action] == 0:
    #             print('zero self transition:', i)
    #             transition_mat[i,i,action] = 0.0001
    
    # handleing termination states
    # transition_mat[255:,:,:] = 0 
    # for i in range(255, 285): 
    #     for j in range(3):
    #         transition_mat[i,112,j] = 1

    if save:
        np.save(path + 'transition_matrix_{}'.format(coef_gravity), arr = transition_mat)
        print('transition matrix saved')
    return transition_mat


# =================================================================================
# from data 
# =================================================================================
def estimate_transition(data): 
    ''' estimate transition matrix from trajectories ''' 
    states = data['state'].to_numpy().astype(int)
    next_states = data['next_state'].to_numpy().astype(int)
    actions = data['action'].to_numpy().astype(int)
    trans_mat = np.zeros((285,285,3))
    for state in range(len(states)): 
        current_state = states[state]
        action = actions[state]
        next_state = next_states[state]
        trans_mat[current_state, next_state, action] += 1

    plane_sum = np.sum(np.sum(trans_mat, axis=0), axis=1)
    for state in range(285):
        if plane_sum[state] == 0:
            trans_mat[state,:,:] = 1/(285*3)
        else: 
            trans_mat[state,:,:] = trans_mat[state,:,:]/plane_sum[state]
    #===Plot===#
    # fig, ax = plt.subplots()
    # im = ax.imshow(trans_mat[:,:,1])
    # plt.show()
    if np.sum(np.isnan(trans_mat)) > 0: 
        print(np.unique(data['episode']), 'nan in transmat')
        exit()
    return trans_mat 

def get_gravity(data):
    ''' calculate gravity from trajectory using environment dynamics''' 
    speeds = data['speed'].to_numpy()
    next_speeds = data['next_speed'].to_numpy()
    positions = data['position'].to_numpy()
    actions = data['action'].to_numpy()

    coef_g = (next_speeds[0] - speeds[0] - (actions[0]-1)*0.001)/(np.cos(3*positions[0]) * -0.0025)
    return coef_g
    

if __name__ == "__main__":
    path = os.path.dirname(os.path.realpath(__file__)) + '/'
    # file_path = '/Users/rw422/Dropbox/SU19_Automation_interaction_IRL/Experiments/Data/'
    # filename = 'Zhide_Wang_MountainCar_results_reshaped.csv'
    # data = pd.read_csv(file_path + filename)
    
    T = sample_transition_matrix(coef_gravity=1.5, save = True)
    T = np.load(path + 'transition_matrix_1.5.npy')
    print(T[17,2,0])
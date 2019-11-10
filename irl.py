# =================================================================================
''' Conditional Choise Probability - Inverse Reinforcement Learning
    Detail see Mohit Sharma 
'''
# =================================================================================
import os, time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np
from environment import obs2state, state2obs
# =================================================================================
#   Written by Ran Wei at rw422@tamu.edu, 7/4/2019
# =================================================================================
def initial_ccp(data): 
    ''' estimate initial ccp from player dataframe. 
        input: 
        state and actions sequence across all example episodes directly from dataframe.
        numpy array for both variables. 
        output: 
        ccp (numpy array) - initial ccp 
    '''
    states = data['state'].to_numpy().astype(int)
    actions = data['action'].to_numpy().astype(int)
    ccp = np.zeros((285, 3))
    for row in range(len(states)): 
        ccp[states[row],actions[row]] += 1
    ccp = ccp/np.sum(ccp, axis = 1).reshape((285,1))
    
    return ccp

def impute_equal(ccp):
    ''' fill in ccp blank entries with equal probabilities for each state.
        input & output: ccp (numpy array) - initial ccp
    '''
    ccp[np.where(np.isnan(ccp) == True)] = 1/3 

    return ccp 

def impute_linear(ccp, data): 
    speed = data['speed'].to_numpy().astype(float)
    actions = data['action'].to_numpy().astype(int)
    actions = (actions == 2).astype(int)

    def estimate_coef(x, y): 
        # number of observations/points 
        n = np.size(x) 
        # mean of x and y vector 
        m_x, m_y = np.mean(x), np.mean(y) 
        # calculating cross-deviation and deviation about x 
        SS_xy = np.sum(y*x) - n*m_y*m_x 
        SS_xx = np.sum(x*x) - n*m_x*m_x 
        # calculating regression coefficients 
        b_1 = SS_xy / SS_xx 
        b_0 = m_y - b_1*m_x 
        return(b_0, b_1)
    coef = estimate_coef(speed, actions)
    idx_miss = np.where(np.isnan(ccp[:,2]))[0]
    prob_right = np.zeros((285,1))
    for state in idx_miss:
        v = state2obs(state)[1]
        prob_right[state,0] = np.clip(coef[0] + coef[1] * v, 0, 1)
    ccp[idx_miss,:] = 0 # clear nan values
    ccp[idx_miss,2] += prob_right[idx_miss,0]
    ccp[idx_miss,0] += (1-prob_right[idx_miss,0])
    ccp[idx_miss,1] = 0
    return ccp

def scale(x, a, b): 
    ''' scale torch array x to range [a,b] '''
    return (x - x.min())/(x.max() - x.min()) * (a - b) + b



def inverse_M(choice_prob, transition_matrix, discount_rate):
    ''' input (all torch tensor): 
        choice_prob - initial choice prob.
    ''' 
    n_state = 285
    choice_prob_repeat = choice_prob.unsqueeze(1).repeat(1, n_state, 1)
    M = torch.eye(n_state).double() - torch.sum(choice_prob_repeat * discount_rate * transition_matrix, dim = 2)
    print(M)
    print(np.linalg.det(M))
    inv_M = M.inverse()

    return inv_M

def perturbation(choice_prob):
    ''' input: 
        choice_prob - initial choice prob (torch tensor).
    ''' 
    choice_prob2 = choice_prob.clone() + 0.0001
    choice_prob2 = choice_prob2 / torch.sum(choice_prob2, dim = 1).view(285,1) # normalize 
    perturb = 0.5772 - torch.log(choice_prob2) 
    return perturb

def update_values(M, shock, reward, choice_prob):   
    ''' input (all torch tensors): 
        M - inverse M square matrix.
        reward - reward parameters.
        choice_prob - updated choice prob.
    ''' 
    reward = reward.repeat(1,3)

    value_update = torch.sum(choice_prob * (reward + shock), dim=1).view(285,1)
    values = M.matmul(value_update)

    return values

def update_policy(values, transition_matrix, reward, sigma, discount_rate): 
    ''' policy = softmax(values). 
        scale values using softmax(x) = softmax(x-c).
        input (all torch tensors).
    '''
    values_adj = (reward + discount_rate * transition_matrix.permute(2, 1, 0).matmul(values))/sigma
    values_adj = values_adj - (values_adj.max() + values_adj.min())/2 # handle overflow
    values_adj = torch.exp(values_adj)
    policy = (values_adj / torch.sum(values_adj, dim = 0)).permute(2,1,0)[0,:,:]
    
    return policy

def nloglikelihood(states, actions, policy): 
    ll = torch.log(policy[states, actions])
    return -ll.sum()

def policy_estimator(data, trans_mat, discount_rate = 0.8, iterations = 1000, theta_lr = 0.1, sigma_lr = 0, disp = False, *arg):
    ''' maximum likelihood policy estimator for conditional choice probability 
        input: 
        data (pandas dataframe) - game record pd dataframe.
        trans_mat (npy file) - environment transition matrix.
        sigma_lr - default sigma_lr is 0 for constant sigma, else set to 0.0001. 
        output: 
        theta (numpy array) - final reward parameters
        sigma (float) - final attention parameter
        values (numpy array) - final state values
        policy (numpy array) - final policy
        ll (list) - list of log likelihood values during learning.
    ''' 
    states = data['state'].to_numpy().astype(int)
    actions = data['action'].to_numpy().astype(int)
    start = time.time()
    # initial parameters setup
    ccp0 = torch.from_numpy(impute_equal(initial_ccp(data)))
    trans_mat = torch.from_numpy(trans_mat)
    inv_M = inverse_M(ccp0, trans_mat, discount_rate)
    shock = perturbation(ccp0)
    # define variables
    # theta = Variable(torch.DoubleTensor(285,1).uniform_(-1,1), requires_grad = True)
    theta_np = -1*np.ones((285,1))
    theta = Variable(torch.DoubleTensor(theta_np), requires_grad = True)
    theta.retain_grad()
    if sigma_lr != 0:
        sigma = Variable(torch.DoubleTensor([1]), requires_grad = True)
        sigma.retain_grad()
    else:
        sigma = 1
    torch.autograd.set_detect_anomaly(True)
    # likelihood gradient descent
    ll,sig = [], []
    ccp = ccp0
    for t in range(iterations): 
        temp_values = update_values(inv_M, shock, theta, ccp)
        temp_policy = update_policy(temp_values, trans_mat, theta, sigma, discount_rate)
        nll = nloglikelihood(states, actions, temp_policy)
        
        try:
            nll.backward()
        except RuntimeError:
            print('ERROR nan values occur in backward calculation')
            break

        if (t + 1)%100 == 0 and disp == True: 
            print('iteration: {},  sigma: {}, negative log likelihood: {}'.\
                format(t+1, round(float(sigma), 2), round(float(-nll), 2)))

        theta.data -= theta_lr * theta.grad.data
        theta.data = scale(theta.data, 1, -1) # bound theta 
        theta.grad.data.zero_()

        if sigma_lr != 0: 
            sigma.data -= sigma_lr * sigma.grad.data
            sigma.grad.data.zero_()
            sig.append(round(float(sigma.data),2))
        ll.append(round(float(-nll), 2))

        if t > 1 and ll[-1] == ll[-2]: # break when ll stop improving
            break

    # display runtime 
    if disp:
        end  = time.time()
        elapse = end - start
        print('total time: ', round(elapse,2))
    # output values to numpy
    theta = theta.data.numpy()
    sigma = float(sigma)
    values = temp_values.data.numpy()
    policy = temp_policy.data.numpy()
    return theta, sigma, values, policy, ll

if __name__ == "__main__":
    from test import reshape_policy, draw_policy, test_policy
    path = os.path.dirname(os.path.realpath(__file__)) + '/'
    file_path = '/Users/rw422/Dropbox/SU19_Automation_interaction_IRL/Experiments/Data/'
    filename = 'Zhide_Wang_MountainCar_results_reshaped.csv'
    data = pd.read_csv(file_path + filename)
    select = [0,60]
    data = data.loc[(data['episode'] >= select[0]) & (data['episode'] <= select[1])]
    trans_mat = np.load(path + 'transition_matrix_1.npy')
    # transition to starting state
    for i in range(255, 285): 
        for j in range(3):
            trans_mat[i,i,j] = 0
            trans_mat[i,112,j] = 1

    theta, sigma, values, policy, ll = policy_estimator(data, trans_mat, sigma_lr=0.0001,iterations = 700, disp = True)
    print(values[112])
    plt.subplot(211)
    plt.plot(ll)
    plt.subplot(212)
    plt.plot(theta)
    plt.show()

# =================================================================================
''' Hidden Rust Model
    For detail see Benjamin Connault 
    Model 1: do not allow hidden transition until episode ends
'''
# =================================================================================
import os, time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =================================================================================
#   Written by Ran Wei at rw422@tamu.edu, 10/25/2019
# =================================================================================
def scale(x, a, b): 
    ''' scale torch array x to range [a,b] '''
    return (x - x.min())/(x.max() - x.min()) * (a - b) + b


def scale_hidden(T_z): 
    ''' scale hidden transition to a stochastic matrix '''
    T_z1 = T_z.clone()
    n_row = T_z1.shape[0]
    for row in range(n_row): 
        if torch.sum(T_z1[row,:] > 0) != n_row or torch.sum(T_z1[row,:] > 0) != 0:
            T_z1[T_z1 < 0] = 0
    T_z1 = T_z1/torch.sum(T_z1, dim=1).view((n_row,1))
    return T_z1


def markov_mat(n): 
    ''' create random markov matrix in numpuy
    n = square matrix dimension '''
    P = np.random.uniform(0,1,size=(n,n))
    P = P/np.sum(P,axis=1).reshape((n,1))
    return P


def rust_transition1(T, T_z):
    ''' 
    construct T_rust with T_environment and T_hidden 
    do not allow hidden transition until episode ends 
    '''
    T_rust = torch.cat( (torch.cat( (T, torch.zeros_like(T)), dim=1 ),\
    torch.cat( (torch.zeros_like(T), T), dim=1 )), dim = 0)
    
    T_rust[255:285,255:285,:] = T_z[0,0] * T[255:285,255:285,:]
    T_rust[255:285,255+285:285+285,:] = T_z[0,1] * T[255:285,255:285,:]
    T_rust[255+285:285+285,255:285,:] = T_z[1,0] * T[255:285,255:285,:]
    T_rust[255+285:285+285,255+285:285+285,:] = T_z[1,1] * T[255:285,255:285,:]
    return T_rust


def end2start(T):
    ''' add end-to-start transition prob = 1 to transition matrix (torch tensor) 
        end state = state > 255 '''
    T1 = T.clone()
    for i in range(255, 285): 
        for j in range(3): # action
            T1[i,112,j] = 1
    return T1


def jacobi_solver(T_e, T_z, u, beta):
    ''' solve for values using Jacobian - Newton's method 
    inputs: 
    pi (N_state x N_state x N_action) - transition matrix, row sum to 1
    u (N_state x N_action) - utility matrix 
    
    intermediates: 
    P_a (N_state x N_action) - action choice probability
    j_v (N_state x N_state) - Jacobian matrix 

    outputs: 
    v (N_state x 1) - value matrix 
    P (N_state x 3) - action choice probability '''
    start = time.time()
    
    T = T_e.clone()
    T[255:, 255:, :] = 0
    T = end2start(T)

    n_state = T.shape[0]*T_z.shape[0]
    n_action = T.shape[2]
    
    T_rust = rust_transition1(T, T_z)

    # initialize 
    j_v = torch.zeros((n_state,n_state,n_action)).double()
    P_a = torch.zeros((n_state, n_action)).double()

    iteration = 30
    v = torch.ones(n_state,1).double() # change to 30 col, each being 1 itr
    for i in range(iteration): 
        for a in range(n_action):
            P_a[:,a] = torch.exp(u[:,a].view(n_state,1) + (beta * T_rust[:,:,a] - torch.eye(n_state).double())\
                .matmul(v.clone())).view((n_state,)) # exponents must be nonpositive
            j_v[:,:,a] = torch.diag(P_a[:,a].view((n_state, ))).matmul(beta * T_rust[:,:,a] - torch.eye(n_state).double()) 

            f_v_sum = torch.sum(P_a, dim=1).view(n_state,1)
            j_v_sum = torch.sum(j_v, dim=2)

        delta_v = torch.solve(-f_v_sum + 1, j_v_sum)[0]
        v += delta_v
        if i == iteration-1: print('final jacobian residual: ', torch.abs(delta_v).max().detach().numpy())
    
    # time 
    end  = time.time()
    print('jacobi solve total time: {} s \n'.format(round(end - start, 2)))
    return v, P_a


def likelihood_filter(df, T_e, T_z, P): 
    ''' 
    calculate likelihood of trajectory using discrete filter 
    only use T_z between episodes 

    inputs: 
    df (N_steps x 3) - player trajectory: episode, state, next_state, action
    T (N_state x N_state x N_action) - transition matrix, add transition to start on top of T_environment

    outputs: 
    -log_p (1x1 torch DoubleTensor) - negative log likelihood of df
    ''' 
    start = time.time()
    T = end2start(T_e)
    T[:,7,:] = 0.01
    T[:, 112,:] = 0.01 

    # create equivalent choice prob matrix for df
    P_choice = torch.zeros((df.shape[0], 2)).double()
    P_choice[:,0] = P[df[:,1], df[:,3]]
    P_choice[:,1] = P[df[:,1] + 285, df[:,3]]
    
    # initialize 
    pi = 0.5*torch.ones(1,2).double()
    log_p = 0
    for i in range(df.shape[0]):
        if i != df.shape[0]-1 : 
            if df[i,0] != df[i+1,0]:
                episode_switch = 1
            else: 
                episode_switch = 0
                
        if episode_switch: 
            pi = pi.matmul(T_z) * T[df[i,1], df[i,2], df[i,3]] * P_choice[i,:]
        else:
            pi = pi * T[df[i,1], df[i,2], df[i,3]] * P_choice[i,:]

        pi_norm = torch.sum(pi)
        pi = pi/pi_norm
        log_p += torch.log(pi_norm)

        if i%1000 == 0:
            print('likelihood filter: step {}, log likelihood {}'.format(i, log_p))

    print('final likelihood: {}, log likelihood {} \n'.format(i, log_p))

    end = time.time()
    print('likelihood solve time: {} s \n'.format(round(end - start,2)))
    return -log_p


def HRM(df, T_e, beta):
    ''' inputs:
        df - (pd dataframe) player trajectory
        beta - discount rate 
        T_e - environment transition matrix
        T_z - transition between hidden states
    ''' 
    
    df = df[['episode', 'state', 'next_state', 'action']].to_numpy()
    # add inter episode transition (connect episodes)
    unique_ep = np.unique(df[:,0])
    counter = 0
    df1 = np.empty([0, 4])
    for ep in unique_ep: 
        df_ep = df[df[:,0] == ep]
        len_ep = df_ep.shape[0]
        counter += len_ep
        insert = np.array([ep, df_ep[df_ep.shape[0]-1,2], 112, 0]).reshape((1,4))
        df_ep = np.concatenate((df_ep, insert), axis = 0)
        df1 = np.concatenate((df1, df_ep), axis = 0)

    # initialize 
    T_z = Variable(torch.from_numpy(markov_mat(2)), requires_grad = True) # 2 hidden states
    T_e = torch.from_numpy(T_e)
    
    n_state = T_e.shape[0]*T_z.shape[0]
    n_action = T_e.shape[2]
    R = Variable(torch.DoubleTensor(n_state,n_action).uniform_(0,1), requires_grad = True)
    
    T_z.retain_grad()
    R.retain_grad()
    torch.autograd.set_detect_anomaly(True)

    # gradient descent 
    lr_R = 0.005
    lr_z = 1e-3
    max_itr = 100

    ll_seq = []
    T_z_seq = []
    P_seq = []
    V_seq = []
    R_seq = []

    HRM_start = time.time()
    for i in range(max_itr):
        print('============================================================================')
        print('iteration {}'.format(i))
        print('============================================================================ \n')
        V, P = jacobi_solver(T_e, T_z, R, beta)
        ll = likelihood_filter(df, T_e, T_z, P)
        print('old transition {} \n'.format(T_z))

        # gradients 
        start = time.time()
        ll.backward()
        R.data -= lr_R * R.grad.data
        T_z.data -= lr_z * T_z.grad.data
        print('R gradient: mean {}, max {}'.format(torch.mean(torch.abs(R.grad.data)), torch.max(torch.abs(R.grad.data))))
        print('transition gradient {} \n'.format(T_z.grad.data))
        
        # scale reward and hidden transition 
        R.data = scale(R.data, 0, 1)
        T_z.data = scale_hidden(T_z.data)
        print('new transition {} \n'.format(T_z))
        
        # check transition matrix markov property
        if torch.abs(torch.sum(T_z, dim = 1) - 1).max() >= 0.00001:
            print(T_z.data)
            print(torch.sum(T_z.data, dim=1))
            print('ERROR: hidden transition not sum to 1 \n')
            break 

        R.grad.data.zero_()
        T_z.grad.data.zero_()
        end  = time.time()

        # collect results 
        ll_seq.append(ll.detach().numpy())
        T_z_seq.append(T_z.detach().data.numpy())
        P_seq.append(P.detach().numpy())
        V_seq.append(V.detach().numpy())
        R_seq.append(R.detach().numpy())

        HRM_current = time.time()
        print('gradient solve time: {} s \n'.format(round(end - start, 2)))
        print('============================================================================')
        print('iteration {}, log-likelihood {} \n'.format(i, np.round(-ll.data.numpy(), 2)))
        print('current time: {} s'.format(round(HRM_current - HRM_start, 2)))
        print('============================================================================ \n')

    ## save results
    np.save(path + 'likelihood', arr = ll_seq)
    np.save(path + 'T_z', arr = T_z_seq)
    np.save(path + 'P', arr = P_seq)
    np.save(path + 'V', arr = V_seq)
    np.save(path + 'R', arr = R_seq)
    print('data saved \n')

    return ll_seq, T_z, P, V, R


if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__)) + '/'
    # x = 0.5 is 255, start is 112
    
    # fig, ax = plt.subplots()
    # im = ax.imshow(trans_mat[:,:,2])
    # plt.show()

    filepath = '/Users/rw422/Dropbox/SU19_Automation_interaction_IRL/Experiments/Data/'
    df_plyer = pd.read_csv(filepath + 'Zhide_Wang_MountainCar_results_reshaped.csv')
    ''' csv episode 1 row 126 transition 2-7-2 = 0, 2-7-a transition problematic '''
    select = [61,105]
    df_plyer = df_plyer.loc[(df_plyer['episode'] >= select[0]) & (df_plyer['episode'] <= select[1])]
    trans_mat = np.load(path + "transition_matrix_2.npy") # check transition matrix is correct
    
    np.random.seed(0)
    torch.manual_seed(0)

    ll, T_z, P, V, R = HRM(df_plyer, trans_mat, 0.995)

    print('hidden transition:', T_z)
    print('reward:', R)

    plt.subplot()
    plt.plot(ll)
    plt.title('HRM log likelihood')
    plt.show()

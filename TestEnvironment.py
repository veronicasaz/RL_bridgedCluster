"""
TestEnvironment: tests simulation environments

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import torch

from env.BridgedCluster_env import Cluster_env

from TrainingFunctions import DQN

from PlotsFunctions import plot_planets_trajectory, plot_planetary_system_trajectory, \
    plot_evolution


def run_trajectory(env, action = 'RL', model_path = None):
    """
    run_trajectory: Run one initialization with RL or with an integrator
    INPUTS:
        action: fixed action or 'RL' 
        env: environment to simulate
        name_suffix: suffix to be added for the file saving
        steps: number of steps to simulate
        reward_f: type of reward to use for the simulation and weights for the 3 terms
        model_path: path to the trained RL algorithm
        steps_suffix: suffix for the file with the steps taken

    OUTPUTS:
        reward: reward for each step
    """
    
    if model_path == None:
        model_path = env.settings['Training']['savemodel'] +'model_weights.pth'
        
    state, info = env.reset()
    i = 0
    terminated = False

    # Case 1: use trained RL algorithm
    if action == 'RL':
         # Load trained policy network
        n_actions = env.action_space.n
        n_observations = env.observation_space_n
        neurons = env.settings['Training']['neurons']
        layers = env.settings['Training']['hidden_layers']

        model = DQN(n_observations, n_actions, neurons, layers)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Take steps
        while terminated == False:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model(state).max(1)[1].view(1, 1)
            state, y, terminated, info = env.step(action.item())
        env.close()
    
    # Case 3: fixed action throughout the simulation
    else:
        while terminated == False:
            x, y, terminated, zz = env.step(action)
            i += 1
        env.close()


def load_state_files(env, namefile = None):
    """
    load_state_files: Load run information 
    INPUTS: 
        env: environment of the saved files
        steps: steps taken
        namefile: suffix for the loading 
    OUTPUTS:
        state: state of the bodies in the system
        cons: action, energy error, angular momentum error
        tcomp: computation time
    """
    env.suffix = (namefile)
    state = env.loadstate()[0]
    cons = env.loadstate()[1]
    tcomp = env.loadstate()[2]

    return state, cons, tcomp


def calculate_errors(states, cons, tcomp):
    cases = len(states)
    steps = np.shape(cons[0][:, 0])[0]

    # Calculate the energy errors
    R = np.zeros((steps, cases))
    E_E = np.zeros((steps, cases))
    E_L = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    Action = np.zeros((steps, cases))
    for i in range(cases):
        R[1:, i] = cons[i][1:steps, 1]
        E_E[1:, i] = abs(cons[i][1:steps, 2]) # absolute relative energy error
        E_L[1:, i] = abs(cons[i][1:steps, 3]) # absolute relative local energy error
        T_c[1:, i] = np.cumsum(tcomp[i][1:steps]) # add individual computation times
        Action[1:, i] = cons[i][1:steps, 0]

    return E_E, E_L, T_c, R, Action

def plot_trajs(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(5, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    legend = True
    plot_traj_index = [0, len(STATES)-1] # plot best and worst
    for case_i, case in enumerate(plot_traj_index): 
        ax1 = fig.add_subplot(gs1[0, case_i])
        ax12 = fig.add_subplot(gs1[1, case_i])
        plot_planets_trajectory(ax1, STATES[case], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)
        plot_planetary_system_trajectory(ax12, STATES[case], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)
        ax1.set_title(Titles[case], fontsize = label_size + 2)
        ax1.set_xlabel('x (au)', fontsize = label_size)
        ax1.set_ylabel('y (au)', fontsize = label_size)
        ax12.set_xlabel('x (au)', fontsize = label_size)
        ax12.set_ylabel('y (au)', fontsize = label_size)
        if case == 0: 
            legend = False
            ax1.legend(loc='upper center', bbox_to_anchor=(0.8, 1.7), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
        

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, Energy_error_local, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(1, len(T_comp), 1)
    ax2 = fig.add_subplot(gs1[2, :])
    ax3 = fig.add_subplot(gs1[3, :])
    ax4 = fig.add_subplot(gs1[4, :])
    for case in range(len(STATES)):
        plot_evolution(ax2, x_axis, Energy_error[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case])
        plot_evolution(ax3, x_axis, Energy_error_local[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case])
        plot_evolution(ax4, x_axis, T_comp[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case])
    
    for ax in [ax2, ax3, ax4]:
        ax.set_yscale('log')

    ax4.set_xlabel('Step', fontsize = label_size)


    ax2.set_ylabel('Energy Error', fontsize = label_size)
    ax3.set_ylabel('Energy Error Local', fontsize = label_size)
    ax4.set_ylabel('Computation time (s)', fontsize = label_size)
    
    ax2.legend(fontsize = label_size -3)

    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_intializations(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,17))
    rows = 4
    columns = 2
    gs1 = matplotlib.gridspec.GridSpec(rows, columns,
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    

    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    legend = True
     # plot best and worst
    for case_i in range(rows): 
        ax1 = fig.add_subplot(gs1[case_i, 0])
        ax12 = fig.add_subplot(gs1[case_i, 1])
        plot_planets_trajectory(ax1, STATES[2*case_i], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], \
                            legend_on = False, axis = 'xy')
        plot_planetary_system_trajectory(ax12, STATES[2*case_i], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'],\
                                  legend_on = False, axis = 'xz')
        ax1.set_title(Titles[2*case_i], fontsize = label_size + 2)
        ax1.set_xlabel('x (au)', fontsize = label_size)
        ax1.set_ylabel('y (au)', fontsize = label_size)
        ax12.set_xlabel('x (au)', fontsize = label_size)
        ax12.set_ylabel('y (au)', fontsize = label_size)
        if case_i == 0: 
            legend = False
            ax1.legend(loc='upper center', bbox_to_anchor=(0.8, 1.7), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
        
    plt.savefig(save_path, dpi = 150)
    plt.show()
        
def plot_rewards(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    # Plot energy error
    linestyle = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
    Energy_error, Energy_error_local, T_comp, R, action= calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(1, len(T_comp), 1)
    ax2 = fig.add_subplot(gs1[0, :])
    ax3 = fig.add_subplot(gs1[1, :])
    ax4 = fig.add_subplot(gs1[2, :])
    ax5 = fig.add_subplot(gs1[3, :])
    for case in range(len(STATES)):
        plot_evolution(ax2, x_axis, Energy_error[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
        plot_evolution(ax3, x_axis, Energy_error_local[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
        plot_evolution(ax4, x_axis, T_comp[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
        plot_evolution(ax5, x_axis, R[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.7), \
                       fancybox = True, ncol = 3, fontsize = label_size-3)

    for ax in [ax2, ax3, ax4, ax5]:
        ax.set_yscale('log')

    ax5.set_xlabel('Step', fontsize = label_size)

    ax2.set_ylabel('Energy Error', fontsize = label_size)
    ax3.set_ylabel('Energy Error Local', fontsize = label_size)
    ax4.set_ylabel('Computation time (s)', fontsize = label_size)
    ax5.set_ylabel('Reward', fontsize = label_size)
    

    plt.savefig(save_path, dpi = 150)
    plt.show()

def calculate_rewards(E_E, E_E_local, T_comp, action, type_reward, W):

    if type_reward == 0:
        a = -(W[0]*(np.log10(abs(E_E[2:]))-np.log10(abs(E_E[1:-1]))) + \
            W[1]*(np.log10(abs(E_E_local[1:])))) +\
            (W[2]*1/abs(np.log10(action)))

    elif type_reward == 1:
        a = -(W[0]*(np.log10(abs(E_E[2:]))-np.log10(abs(E_E[1:-1]))) + \
            W[1]*(np.log10(abs(E_E_local[2:]))-np.log10(abs(E_E_local[1:-1])))) +\
            (W[2]*1/abs(np.log10(action)))
    elif type_reward == 2:
        a = -(W[0]*np.log10(abs(E_E[1:])) + \
            W[1]*(np.log10(abs(E_E_local[1:])))) +\
            (W[2]*1/abs(np.log10(action)))

    return a

def plot_reward_comparison(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))

    reward_types = 6
    gs1 = matplotlib.gridspec.GridSpec(4, 1, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    # Plot energy error
    linestyle = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
    Energy_error, Energy_error_local, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(1, len(T_comp), 1)

    W = np.zeros([[1.0, 1.0, 1e-2],
                  [1.0, 10.0, 1e-2],
                  [1.0, 1.0, 1e-2],
                  [1.0, 10.0, 1e-2],
                  [1.0, 1.0, 1e-2]
                  ])
    
    ax1 = fig.add_subplot(gs1[0, :])
    ax2 = fig.add_subplot(gs1[1, :])
    ax3 = fig.add_subplot(gs1[2, :])
    
    plot_evolution(ax1, x_axis, Energy_error[1:, 0], label = Titles[0][1:], \
                       colorindex = 0, linestyle = linestyle[0])
    plot_evolution(ax2, x_axis, Energy_error_local[1:, case], label = Titles[case][1:], \
                       colorindex = 0, linestyle = linestyle[0])
    plot_evolution(ax3, x_axis, T_comp[1:, 0], label = Titles[0][1:], \
                       colorindex = 0, linestyle = linestyle[0])
    
    Rewards = np.zeros(reward_types)
    ax4 = fig.add_subplot(gs1[3, :])
    for case in range(reward_types):
        Rewards[case] = calculate_rewards(Energy_error, Energy_error_local, T_comp, \
                                    action, case, W[case, :])
        plot_evolution(ax4, x_axis, Rewards[case], \
                       label = 'Reward %i, W = %i, %i, %i'%(case, W[case, 0], W[case, 1], W[case, 2]), \
                       colorindex = case//2, linestyle = linestyle[case%2])
        ax4.set_yscale('log')
        ax4.set_ylabel('Reward', fontsize = label_size)
    
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.7), \
                       fancybox = True, ncol = 3, fontsize = label_size-3)
    ax4.set_xlabel('Step', fontsize = label_size)

    plt.savefig(save_path, dpi = 150)
    plt.show()

if __name__ == '__main__':
    experiment = 1 # number of the experiment to be run
            
    if experiment == 1: # run bridge for all actions
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '1_run_actions/'
        env.settings['InitialConditions']['seed'] = 1

        NAMES = []
        # for act in range(1):
        for act in range(env.settings['RL']['number_actions']):
            name = '_action'+str(act)
            NAMES.append(name)
            env.settings['Integration']['suffix'] = NAMES[act]
            run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        # for act in range(1):
        for act in range(env.settings['RL']['number_actions']):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison.png'
        plot_trajs(env, STATE, CONS, TCOMP, NAMES, save_path)
    
    elif experiment == 2: # test different initializations
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '2_run_initializations/'

        NAMES = []
        seeds = [0, 1, 2, 3]

        for i in range(len(seeds)):
            for j in [0, env.settings['RL']['number_actions']-1]: # best and worst actions
                name = '_seed'+ str(seeds[i])+'_action'+str(j)
                NAMES.append(name)
                env.settings['Integration']['suffix'] = name
                env.settings['InitialConditions']['seed'] = seeds[i]
                # run_trajectory(env, action = j) # Choose an action in between

        STATE = []
        CONS = []
        TCOMP = []
        for i in range(len(NAMES)):
            env.settings['Integration']['suffix'] = NAMES[i]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Initialization_comparison.png'
        plot_intializations(env, STATE, CONS, TCOMP, NAMES, save_path)
        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Reward_comparison.png'
        plot_rewards(env, STATE, CONS, TCOMP, NAMES, save_path)

    elif experiment == 3: # test rewards
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '3_compare_rewards/'
        env.settings['InitialConditions']['seed'] = 1

        env2 = Cluster_env()
        env2.settings['Integration']['subfolder'] = '1_run_actions/'
        STATE = []
        CONS = []
        TCOMP = []
        NAMES = []
        for act in range(env.settings['RL']['number_actions']):
            name = '_action'+str(act)
            NAMES.append(name)
        for act in range(env.settings['RL']['number_actions']):
            env2.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env2)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        action = 2

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Reward_comparison.png'
        plot_reward_comparison(env, STATE[action], CONS[action], TCOMP[action], NAMES[action], save_path)






        
        
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

from ENVS.bridgedparticles.envs.Bridged2Body_env import TwoBody_env
from ENVS.bridgedparticles.envs.Bridged3Body_env import ThreeBody_env
from ENVS.bridgedparticles.envs.BridgedCluster_env import BridgedCluster_env
from TrainingFunctions import DQN

from PlotsFunctions import plot_planets_trajectory, plot_evolution


def run_trajectory(env, action = 'RL', model_path = None, architecture = None):
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
        
    state, info = env.reset(seed = seed, steps = steps, typereward = reward_type)

    reward = np.zeros(steps)
    i = 0

    # Case 1: use trained RL algorithm
    if action == 'RL':
         # Load trained policy network
        n_actions = env.action_space.n
        n_observations = len(state)
        if architecture != None:
            model = DQN(n_observations, n_actions, settings = env.settings, \
                        layers = architecture[0], neurons = architecture[1])
        else:
            model = DQN(n_observations, n_actions, settings = env.settings) # we do not specify ``weights``, i.e. create untrained model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        steps_taken = list()
        steps_taken.append(0) # initial conditions

        # Take steps
        while terminated == False:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model(state).max(1)[1].view(1, 1)
            steps_taken.append(action.item())
            state, y, terminated, info = env.step(action.item())
            reward[i] = env.reward
        path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'reward_' + env.settings['Integration']['suffix']
        np.save(path, np.array(reward))
        np.save(env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'RL_steps_taken_'  + env.settings['Integration']['suffix'], np.array(steps_taken))
        env.close()
    
    # Case 3: fixed action throughout the simulation
    else:
        while terminated == False:
            x, y, terminated, zz = env.step(action)
            reward[i] = env.reward
            i += 1
        path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'reward_' + env.settings['Integration']['suffix']
        np.save(path, np.array(reward))
        env.close()
    return reward


def load_state_files(env, steps, namefile = None):
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
    state = env.loadstate()[0][0:steps, :, :]
    cons = env.loadstate()[1][0:steps, :]
    tcomp = env.loadstate()[2][0:steps]

    return state, cons, tcomp


def calculate_errors(states, cons, tcomp, steps):
    cases = len(states)

    # Calculate the energy errors
    E_E = np.zeros((steps, cases))
    E_M = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    for i in range(cases):
        E_E[1:, i] = abs(cons[i][1:steps, 1]) # absolute relative energy error
        E_M[1:, i] = np.linalg.norm((cons[i][1:steps, 2:] - cons[i][0:steps-1, 2:]), axis = 1) # relative angular momentum error    
        T_c[1:, i] = np.cumsum(tcomp[i][1:steps]) # add individual computation times

    return E_E, T_c


def plot_trajs(env, STATES, CONS, Titles, filenames, save_path = None):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,10))
    gs1 = matplotlib.gridspec.GridSpec(3, len(STATES), width_ratios=[1]*len(STATES), 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    legend = True
    for case in range(len(STATES)):
        ax1 = fig.add_subplot(gs1[0, case])
        plot_planets_trajectory(ax1, STATES[case]/1.496e11, name_bodies, \
                            labelsize=label_size, steps = steps, legend_on = False)
        ax1.set_title(Titles[case], fontsize = label_size + 2)
        ax1.set_xlabel('x (au)', fontsize = label_size)
        ax1.set_ylabel('y (au)', fontsize = label_size)
        if case == 1: 
            # legend = False
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
        

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    x_axis = np.arange(0, steps-1, 1)
    Energy_error, T_comp = calculate_errors(env, len(STATES), steps, filenames)
    print(np.shape(Energy_error), np.shape(T_comp))
    ax2 = fig.add_subplot(gs1[1, :])
    ax3 = fig.add_subplot(gs1[2, :])
    for case in range(len(STATES)):
        plot_evolution(ax2, x_axis, Energy_error[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case])
        plot_evolution(ax3, x_axis, T_comp[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case])
        ax2.set_xlabel('Step', fontsize = label_size)
        ax3.set_xlabel('Step', fontsize = label_size)
        ax2.set_ylabel('Energy Error', fontsize = label_size)
        ax3.set_ylabel('Computation time (s)', fontsize = label_size)
    
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    ax2.legend(fontsize = label_size -3)

    plt.savefig(save_path + "state_bridgevsNobridge.png", dpi = 150)
    plt.show()

if __name__ == '__main__':
    experiment = 1 # number of the experiment to be run
            
    if experiment == 1: # run bridge for all actions
        
        steps = 100

        env = BridgedCluster_env()
        env.settings['Integration']['subfolder'] = '1_run_actions/'

        NAMES = []
        for act in range(env.settings['RL']['number_actions']):
            NAMES.append('action'+ str(env.actions[act]))
            env.settings['Integration']['suffix'] = NAMES[act]
            run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        for act in range(env.settings['RL']['number_actions']):
            state, cons, tcomp = load_state_files(env, namefile = NAMES[act])
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison.png'
        plot_trajs(env, STATE, CONS, TCOMP, NAMES, save_path)
        



        
        
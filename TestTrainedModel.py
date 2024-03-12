"""
TestTrainedModelGym_hermite: tests and plots for the RL algorithm

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import json
import seaborn as sns

import torch
import torchvision.models as models
import gym

from  ENVS.bridgedparticles.envs.Bridged3Body_env import ThreeBody_env
from TrainingFunctions import DQN, load_reward, plot_reward
from PlotsFunctions import load_state_files, plot_planets_trajectory, \
                            run_trajectory, plot_evolution,\
                            plot_actions_taken, plot_planets_distance,\
                            calculate_errors


colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']


def plot_trajs(env, STATES, CONS, TCOMP,  Titles, filenames, steps_taken = None, save_path = None, steps = 100):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    columns = 4
    rows = (len(STATES)//columns + 4)*2
    gs1 = matplotlib.gridspec.GridSpec(rows, columns, 
                                    left=0.1, wspace=0.4, hspace = 1.5, right = 0.93,
                                    top = 0.9, bottom = 0.04)
    
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    legend = True
    AX = []
    for case in range(len(STATES)):
        column = 2*(case//columns)
        ax1 = fig.add_subplot(gs1[column:column+2, case%columns])
        print(column, case%columns)
        plot_planets_trajectory(ax1, STATES[case]/1.496e11, name_bodies, \
                            labelsize=label_size, steps = steps, \
                                legend_on = False, axislabel_on = False)
        ax1.set_title(Titles[case], fontsize = label_size + 2)
        if case//3 == 1:
            ax1.set_xlabel('x (au)', fontsize = label_size)
        if case%3 == 0:
            ax1.set_ylabel('y (au)', fontsize = label_size)
        
        if case == 1: 
            # legend = False
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.8), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
        AX.append(ax1)

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    x_axis = np.arange(0, steps, 1)
    column = 2*(len(STATES)//columns)
    Energy_error, T_comp = calculate_errors(STATES, CONS, TCOMP, steps)
    ax2 = fig.add_subplot(gs1[column:column+2, :-1])
    ax3 = fig.add_subplot(gs1[column+2:column+4, :-1])
    ax4 = fig.add_subplot(gs1[column+4:column+6, :-1])
    AX.append(ax2)
    AX.append(ax3)
    AX.append(ax4)
    for case in range(len(STATES)):
        plot_evolution(ax2, x_axis[1:], Energy_error[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case])
        plot_evolution(ax3, x_axis[1:], T_comp[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case])

    ax2.set_ylabel('Energy Error', fontsize = label_size)
    ax3.set_ylabel('Computation time (s)', fontsize = label_size)
    
    plot_planets_distance(ax4, x_axis[1:], STATES[0][1:], name_bodies, steps = steps, legend= False)
    ax4.set_ylabel('Distance (au)', fontsize = label_size)
    
    
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')

    # ax3.legend(fontsize = label_size -3, loc = 'upper left')
    ax2.legend(loc='upper center', bbox_to_anchor=(1.2, 1), \
                       fancybox = True, ncol = 1, fontsize = label_size-3)
    ax4.legend(loc='upper center', bbox_to_anchor=(1.2, 1), \
                       fancybox = True, ncol = 1, fontsize = label_size-3)

    # Plot close encounters
 

    # Plot actions of RL
    ax5 = fig.add_subplot(gs1[column+6:column+8, :-1])
    AX.append(ax5)
    if np.any(steps_taken) == None:
        steps_taken = np.load(env.settings['Integration']['savefile'] + 'RL_steps_taken.npy', allow_pickle=True)
    plot_actions_taken(ax5, x_axis[1:], steps_taken[1:]) # the first action is fixed
    ax5.set_yticks(np.arange(0, env.settings['Integration']['number_actions']**2))
    ax5.set_xlabel('Step', fontsize = label_size)

    # General settings
    for ax_i in AX:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)

    plt.savefig(save_path + "state_bridgevsNobridge.png", dpi = 150)
    # plt.show()

def plot_trajs_together(env, STATES, CONS, Titles, filenames, save_path = None):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(6, 2, 
                                    left=0.1, wspace=0.4, hspace = 1.5, right = 0.93,
                                    top = 0.9, bottom = 0.04)
    
    markers = ['o', 'x', 's']
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    legend = True
    AX = []
    ax1 = fig.add_subplot(gs1[0:2, 0])
    AX.append(ax1)

    print(len(STATES))
    for case in range(len(STATES)):
        plot_planets_trajectory(ax1, STATES[case]/1.496e11, name_bodies, \
                            labelsize=label_size, steps = steps, \
                                legend_on = False, axislabel_on = False, marker = markers[case])
    ax1.set_title(Titles[case], fontsize = label_size + 2)
    ax1.set_xlabel('x (au)', fontsize = label_size)
    ax1.set_ylabel('y (au)', fontsize = label_size)       
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.8), \
                    #    fancybox = True, ncol = 3, fontsize = label_size-2)

    # Plot energy error
    linestyle = '-'
    x_axis = np.arange(0, steps, 1)
    columns = 2
    Energy_error, T_comp = calculate_errors(env, len(STATES), steps, filenames)
    ax2 = fig.add_subplot(gs1[columns:columns+2, :])
    ax3 = fig.add_subplot(gs1[columns+2:columns+4, :])
    AX.append(ax2)
    AX.append(ax3)
    for case in range(len(STATES)):
        plot_evolution(ax2, x_axis[1:], Energy_error[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle)
        plot_evolution(ax3, x_axis[1:], T_comp[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle)
        # ax2.set_xlabel('Step', fontsize = label_size)
        # ax3.set_xlabel('Step', fontsize = label_size)
        ax2.set_ylabel('Energy Error', fontsize = label_size)
        ax3.set_ylabel('Computation time (s)', fontsize = label_size)
    
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax3.legend(fontsize = label_size -8, loc = 'upper left')

    # General settings
    for ax_i in AX:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)

    plt.savefig(save_path + "state_bridgevsNobridge_together.png", dpi = 150)
    plt.show()

def run_bridge_vs_no_bridge(env, seed, steps):
    """
    run_bridge_vs_no_bridge: run commands to run trajectories and save data
    """
    env.bridged = False
    env.integrator = 'Hermite'
    env.t_step_integr = [1e-2]
    run_trajectory(seed = seed, action = 5, env = env,\
                            name_suffix = '_nobridge_Hermite', steps = steps)
    
    print("------------------------------------")
    env.bridged = False
    env.integrator = 'Huayno'
    env.t_step_integr = [1e-2]
    run_trajectory(seed = seed, action = 0, env = env,\
                            name_suffix = '_nobridge_Huayno', steps = steps)
    
    print("------------------------------------")
    env.bridged = True
    env.integrator = 'Hermite'
    run_trajectory(seed = seed, action = 0, env = env,\
                            name_suffix = '_bridge_b0_i0', steps = steps)
    
    print("------------------------------------")
    env.bridged = True
    env.integrator = 'Hermite'
    run_trajectory(seed = seed, action = 3, env = env,\
                            name_suffix = '_bridge_b3_i0', steps = steps)
    
    print("------------------------------------")
    env.bridged = True
    env.integrator = 'Hermite'
    run_trajectory(seed = seed, action = 10, env = env,\
                            name_suffix = '_bridge_b2_i2', steps = steps)
    
    print("------------------------------------")
    env.bridged = True
    env.integrator = 'Hermite'
    run_trajectory(seed = seed, action = 12, env = env,\
                            name_suffix = '_bridge_b0_i3', steps = steps)
    
    print("------------------------------------")
    env.bridged = True
    env.integrator = 'Hermite'
    run_trajectory(seed = seed, action = 15, env = env,\
                            name_suffix = '_bridge_b3_i3', steps = steps)
    
    # print("------------------------------------")
    env.bridged = True
    env.integrator = 'Hermite'
    run_trajectory(seed = seed, action = 'RL', env = env,\
                            name_suffix = '_bridge_RL', steps = steps)

    
def prepare_plot_bridge_vs_no_bridge(env, steps):
        """
        plot_bridge_vs_no_bridge: load files and call plot
        """
        state_nobridge_Hermite, cons_nobridge_Hermite, tcomp_nobridge_Hermite = load_state_files(env, steps, namefile = '_nobridge_Hermite')
        state_nobridge_Huayno, cons_nobridge_Huayno, tcomp_nobridge_Huayno = load_state_files(env, steps, namefile = '_nobridge_Huayno')
        state_bridge_00, cons_bridge_00, tcomp_bridge_00 = load_state_files(env, steps, namefile = '_bridge_b0_i0')
        state_bridge_30, cons_bridge_30, tcomp_bridge_30 = load_state_files(env, steps, namefile = '_bridge_b3_i0')
        state_bridge_22, cons_bridge_22, tcomp_bridge_22 = load_state_files(env, steps, namefile = '_bridge_b2_i2')
        state_bridge_03, cons_bridge_03, tcomp_bridge_03 = load_state_files(env, steps, namefile = '_bridge_b0_i3')
        state_bridge_33, cons_bridge_33, tcomp_bridge_33 = load_state_files(env, steps, namefile = '_bridge_b3_i3')
        state_bridge_RL, cons_bridge_RL, tcomp_bridge_RL = load_state_files(env, steps, namefile = '_bridge_RL')
        
        path_save = env.settings["Integration"]['savefile'] + env.subfolder
        plot_trajs(env,\
                   [state_nobridge_Hermite, state_nobridge_Huayno, state_bridge_00, state_bridge_30, state_bridge_22,state_bridge_03, state_bridge_33, state_bridge_RL], 
                   [cons_nobridge_Hermite, cons_nobridge_Huayno, cons_bridge_00, cons_bridge_30, cons_bridge_22,cons_bridge_03, cons_bridge_33, cons_bridge_RL], 
                   [tcomp_nobridge_Hermite, tcomp_nobridge_Huayno, tcomp_bridge_00, tcomp_bridge_30, tcomp_bridge_22,tcomp_bridge_03, tcomp_bridge_33, tcomp_bridge_RL],
                    ['No bridge Hermite', 'No bridge Huayno', r'Bridge 00', r'Bridge 30', r'Bridge 22', r'Bridge 03', r'Bridge 33', r'Bridge RL'],\
                    ['_nobridge_Hermite', '_nobridge_Huayno', '_bridge_b0_i0','_bridge_b3_i0','_bridge_b2_i2','_bridge_b0_i3','_bridge_b3_i3', '_bridge_RL'], save_path = path_save, 
                    steps = steps)

def prepare_plot_together(env, steps):
        """
        plot_bridge_vs_no_bridge: load files and call plot
        """
        state_nobridge_Hermite, cons_nobridge_Hermite, tcomp_nobridge_Hermite = load_state_files(env, steps, namefile = '_nobridge_Hermite')
        state_nobridge_Huayno, cons_nobridge_Huayno, tcomp_nobridge_Huayno = load_state_files(env, steps, namefile = '_nobridge_Huayno')
        state_bridge_fast, cons_bridge_fast, tcomp_bridge_fast = load_state_files(env, steps, namefile = '_bridge_fast')
        state_bridge_accurate, cons_bridge_accurate, tcomp_bridge_accurate = load_state_files(env, steps, namefile = '_bridge_accurate')
        state_bridge_mid, cons_bridge_mid, tcomp_bridge_mid = load_state_files(env, steps, namefile = '_bridge_mid')
        state_bridge_RL, cons_bridge_RL, tcomp_bridge_RL = load_state_files(env, steps, namefile = '_bridge_RL')
        
        path_save = env.settings["Integration"]['savefile'] + env.subfolder
        plot_trajs_together(env,\
                   [state_nobridge_Hermite, state_bridge_accurate], 
                   [], ['No bridge Hermite', r'Bridge $10^{-4}$'],\
                    ['_nobridge_Hermite', '_bridge_accurate'], save_path = path_save)
        
        
if __name__ == '__main__':
    experiment = 2 # number of the experiment to be run
    seed = 1

    if experiment == 1:
        # Plot training results
        a = ThreeBody_env()
        reward, EnergyError, HuberLoss = load_reward(a)
        plot_reward(a, reward, EnergyError, HuberLoss)
        
    elif experiment == 2:
        # particles 4 and 5 are the planets around
        env = ThreeBody_env()
        env.subfolder = '2_TrainingResults/'

        steps = 150

        run_bridge_vs_no_bridge(env, seed, steps)
        prepare_plot_bridge_vs_no_bridge(env, steps)
        # prepare_plot_together(env, steps)



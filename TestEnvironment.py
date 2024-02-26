"""
TestEnvironment: tests simulation environments

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from ENVS.bridgedparticles.envs.Bridged2Body_env import TwoBody_env
from ENVS.bridgedparticles.envs.Bridged3Body_env import ThreeBody_env
from PlotsFunctions import run_trajectory, load_state_files, \
                            plot_planets_trajectory, plot_evolution

def calculate_errors(env, cases, steps, namefile):
    state = list()
    cons = list()
    tcomp = list()
    name = list()
    for i in range(cases):
        state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = namefile[i])
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        name.append('Case_%i'%i)

    # Calculate the energy errors
    E_E = np.zeros((steps, cases))
    E_M = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    for i in range(cases):
        E_E[1:, i] = abs(cons[i][1:, 1]) # absolute relative energy error
        E_M[1:, i] = np.linalg.norm((cons[i][1:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error
        T_c[:, i] = np.cumsum(tcomp[i]) # add individual computation times

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
    linestyle = '-'
    x_axis = np.arange(0, steps-1, 1)
    Energy_error, T_comp = calculate_errors(env, len(STATES), steps, filenames)
    print(np.shape(Energy_error), np.shape(T_comp))
    ax2 = fig.add_subplot(gs1[1, :])
    ax3 = fig.add_subplot(gs1[2, :])
    for case in range(len(STATES)):
        plot_evolution(ax2, x_axis, Energy_error[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle)
        plot_evolution(ax3, x_axis, T_comp[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle)
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
    experiment = 3 # number of the experiment to be run
    seed = 1

    def run_bridge_vs_no_bridge(env, seed, steps):
        """
        run_bridge_vs_no_bridge: run commands to run trajectories and save data
        """
        env.bridged = True
        env.integrator = 'Hermite'
        env.t_step_integr = [1e-2, 1e-2] # smaller time-step parameter for the one with the planets
        run_trajectory(seed = seed, action = 5, env = env,\
                               name_suffix = '_bridge_fast', steps = steps)
        

        env.bridged = True
        env.integrator = 'Hermite'
        env.t_step_integr = [1e-2, 1e-2] # smaller time-step parameter for the one with the planets
        run_trajectory(seed = seed, action = 0, env = env,\
                               name_suffix = '_bridge_accurate', steps = steps)
        
        env.bridged = False
        env.integrator = 'Hermite'
        env.t_step_integr = 1e-2
        run_trajectory(seed = seed, action = 0, env = env,\
                               name_suffix = '_nobridge_Hermite', steps = steps)
        
        env.bridged = False
        env.integrator = 'Huayno'
        env.t_step_integr = 1e-2
        run_trajectory(seed = seed, action = 0, env = env,\
                               name_suffix = '_nobridge_Huayno', steps = steps)
    
    def plot_bridge_vs_no_bridge(env, steps):
        """
        plot_bridge_vs_no_bridge: load files and call plot
        """
        state_bridge_fast, cons_bridge_fast, tcomp_bridge_fast = load_state_files(env, steps, namefile = '_bridge_fast')
        state_bridge_accurate, cons_bridge_accurate, tcomp_bridge_accurate = load_state_files(env, steps, namefile = '_bridge_accurate')
        state_nobridge_Hermite, cons_nobridge_Hermite, tcomp_nobridge_Hermite = load_state_files(env, steps, namefile = '_nobridge_Hermite')
        state_nobridge_Huayno, cons_nobridge_Huayno, tcomp_nobridge_Huayno = load_state_files(env, steps, namefile = '_nobridge_Huayno')
        
        path_save = env.settings["Integration"]['savefile'] + env.subfolder
        plot_trajs(env, \
                   [state_nobridge_Hermite, state_nobridge_Huayno, state_bridge_fast, state_bridge_accurate], 
                   [], ['No bridge Hermite', 'No bridge Huayno', r'Bridge $10^{-2}$', r'Bridge $10^{-4}$'],\
                    ['_nobridge_Hermite', '_nobridge_Huayno', '_bridge_fast', '_bridge_accurate'], save_path = path_save)
        
    if experiment == 1: # run bridged vs not bridged for 2 particles
        
        steps = 100

        env = TwoBody_env()
        env.n_bodies = 2 # overwrite to 3 bodies
        env.subfolder = '1_runBridgedvsNobridge_2BP/'
        env.t_step_integr = 1e-3
        env.t_step_bridge = 1e-3

        run_bridge_vs_no_bridge(env, seed, steps)
        plot_bridge_vs_no_bridge(env, steps)
        
    elif experiment == 2: # run bridged vs not bridged for 3 particles
        steps = 30

        env = TwoBody_env()
        env.n_bodies = 3 # overwrite to 3 bodies
        env.subfolder = '2_runBridgedvsNobridge_3BP/'
        env.t_step_integr = 1e-3
        env.t_step_bridge = 1e-3

        run_bridge_vs_no_bridge(env, seed, steps)
        plot_bridge_vs_no_bridge(env, steps)
    
    elif experiment == 3: # run bridged vs not bridged for 3 particles with planets
        steps = 100

        # particles 4 and 5 are the planets around
        env = ThreeBody_env()
        env.settings['Integration']['n_bodies'] = 3 # overwrite to 3 bodies
        env.subfolder = '3_runBridgedvsNobridge_planetary/'
        env.bodies_inner = [0, 0, 2]

        run_bridge_vs_no_bridge(env, seed, steps)
        plot_bridge_vs_no_bridge(env, steps)



        
        
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
from PlotsFunctions import run_trajectory, load_state_files, \
                            plot_planets_trajectory

def plot_trajs(STATES, CONS, Titles):

    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,10))
    gs1 = matplotlib.gridspec.GridSpec(2, len(STATES), width_ratios=[1]*len(STATES), 
                                    left=0.08, wspace=0.2, hspace = 0.5, right = 0.93,
                                    top = 0.97, bottom = 0.01)
    
    
    # name_bodies = (np.arange(np.shape(STATES[[0]])[1])+1).astype(str)
    name_bodies = ['1', '2', '3']
    legend = True
    for case in range(len(STATES)):
        ax = fig.add_subplot(gs1[0, case])
        if case > 0: 
            legend = False
        plot_planets_trajectory(ax, STATES[case]/1.496e11, name_bodies, \
                            labelsize=label_size, steps = steps, legend_on = legend)
        ax.set_title(Titles[case], fontsize = label_size + 2)

    plt.show()

if __name__ == '__main__':
    experiment = 1 # number of the experiment to be run
    seed = 1

    if experiment == 1: # run bridged vs not bridged for 2 particles
        t_step = 1e-4
        steps = 100

        env = TwoBody_env()
        env.settings['Integration']['n_bodies'] = 2 # overwrite to 2 bodies
        env.subfolder = '1_runBridgedvsNobridge/'

        env.settings['Integration']['bridged'] = True
        env.reset(seed = seed, steps = steps)
        run_trajectory(seed = seed, action = 0, env = env,\
                               name_suffix = '_bridge', steps = steps)

        env.settings['Integration']['bridged'] = False
        env.reset(seed = seed, steps = steps)
        run_trajectory(seed = seed, action = 0, env = env,\
                               name_suffix = '_nobridge', steps = steps)
        
     
        
        env.close()

        # plot comparison
        state_nobridge, cons_nobridge, tcomp_nobridge = load_state_files(env, steps, namefile = '_nobridge')
        state_bridge, cons_bridge, tcomp_bridge = load_state_files(env, steps, namefile = '_bridge')
        plot_trajs([state_nobridge, state_bridge], [], ['No bridge', 'Bridge'])


        
        
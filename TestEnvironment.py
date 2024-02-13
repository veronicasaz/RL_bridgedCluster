"""
TestEnvironment: tests simulation environments

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from ENVS.envs_2D.envs.Bridged2Body_env import TwoBody_env
from PlotsFunctions import run_trajectory, load_state_files, \
                            plot_planets_trajectory

def plot_trajs(STATES, CONS):

    # Setup plot
    label_size = 18
    fig, ax = plt.subplots(nrows=3, ncols= 1, layout = 'constrained', figsize = (8, 12))
    
    name_bodies = (np.arange(np.shape(STATES[[0]])[1])+1).astype(str)
    for case in range(len(STATES)):
        plot_planets_trajectory(ax[0], STATES[case]/1.496e11, name_bodies, \
                            labelsize=label_size, steps = steps, legend_on = True)

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

        env.settings['Integration']['bridged'] = False
        env.reset(seed = seed, steps = steps)
        run_trajectory(seed = seed, action = t_step, env = env,\
                               name_suffix = 'nobridge', steps = steps)
        
        env.settings['Integration']['bridged'] = True
        env.reset(seed = seed, steps = steps)
        run_trajectory(seed = seed, action = t_step, env = env,\
                               name_suffix = 'bridge', steps = steps)
        
        env.close()

        # plot comparison
        state_nobridge, cons_nobridge, tcomp_nobridge = load_state_files(env, steps, namefile = 'nobridge')
        state_bridge, cons_bridge, tcomp_bridge = load_state_files(env, steps, namefile = 'bridge')


        
        
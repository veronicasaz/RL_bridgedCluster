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

from env_hydro.BridgedCluster_env_hydro import Cluster_env_hydro
from TestEnvironment import run_trajectory, load_state_files, plot_trajs
from Plots_TestEnvironment import plot_comparison_end, plot_distance_action, \
    plot_energy_vs_tcomp, plot_energy_vs_tcomp_avg, plot_energy_vs_tcomp_avg_together,\
    plot_a_vs_e, plot_trajs_hydro



colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']

if __name__ == '__main__':
    experiment = 0 # number of the experiment to be run

    # import env
    # env = gym.make('Cluster_env_hydro-v0')

    if experiment == 0:
        # 2: use network trained with hyperparameters chosen by hand
        env = Cluster_env_hydro()
        env.settings['Integration']['subfolder'] = '11_run_hydro_traj/'

        model_path_index = '173'
        model_path = './Training_Results/model_weights'+model_path_index +'.pth'
        index_to_plot = [0,3, 9]
        # index_to_plot = [0]
        
        NAMES = []
        TITLES = []
        NAMES.append('_actionRL')
        env.settings['Integration']['suffix'] = NAMES[0]
        TITLES.append(r'RL-173')
        run_trajectory(env, action = 'RL', model_path= model_path)
        # env.plot_orbit()
        
        for act_i, act in enumerate(index_to_plot):
            # act = act+9
            NAMES.append('_action_%0.3E'%(env.actions[act]))
            env.settings['Integration']['suffix'] = NAMES[act_i+1]
            TITLES.append(r'%i: $\Delta t_B$ = %.1E'%(act, env.actions[act]))
            print(env.actions[act])
            run_trajectory(env, action = act)
            # env.plot_orbit()

        STATE = []
        CONS = []
        TCOMP = []
        TITLES2 = []
        # index_to_plot = [0,10]
        for act in range(len(NAMES)):
        # for act in range(4):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)
            TITLES2.append(TITLES[act])

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison_RL.png'
        plot_trajs_hydro(env, STATE, CONS, TCOMP, TITLES2, save_path, plot_traj_index=[0,1])
        # save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            # 'Action_comparison_RL2.png'
        # plot_state_diff(env, STATE, CONS, TCOMP, TITLES, save_path)
        # save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            # 'Action_comparison_RL3.png'
        # plot_comparison_end(env, STATE, CONS, TCOMP, NAMES, save_path, plot_traj_index=[0,1])

    elif experiment == 3:
        # 2: use network trained with hyperparameters chosen by hand
        env = Cluster_env()
        env.settings['Training']['RemovePlanets'] = False

        env.settings['Integration']['max_error_accepted'] = 1e5
        env.settings['Integration']['max_steps'] = 40

        seed = 2
        hybrid = False
        env.settings['InitialConditions']['seed'] = seed
        env.settings['InitialConditions']['bodies_in_system'] = 'fixed'

        model_path_index = '173'
        model_path = './Training_Results/model_weights'+model_path_index +'.pth'
        index_to_plot = [0,2,5, 7, 9]
        
        Bodies = [5, 9, 15]
        for i in range(len(Bodies)):
            env.settings['InitialConditions']['n_bodies'] = Bodies[i]
            env.settings['Integration']['subfolder'] = '42_runmodel_Nvary/%ibodies/'%Bodies[i]
            NAMES = []
            TITLES = []

            # RL
            NAMES.append('_actionRL_seed%i'%seed)
            env.settings['Integration']['suffix'] = NAMES[0]
            TITLES.append(r"RL-"+model_path_index)
            env.settings['Integration']["hybrid"] = False
            # run_trajectory(env, action = 'RL', model_path= model_path)

            if hybrid == True:
            # Hybrid
                NAMES.append('_actionHRL_seed%i'%seed)
                env.settings['Integration']['suffix'] = NAMES[1]
                TITLES.append(r"H-RL-"+model_path_index)
                env.settings['Integration']["hybrid"] = True
                # run_trajectory(env, action = 'RL', model_path= model_path)
                traj_index = 'HRLbest'
            else:
                traj_index = 'RLbest'

            # Fixed-step
            for a_i, act in enumerate(index_to_plot): # index_toplot includes RL
                action = act
                NAMES.append('_action_%0.3E_seed%i'%(env.actions[action], seed))
                env.settings['Integration']['suffix'] = NAMES[a_i]
                TITLES.append(r'%i: $\Delta t_B$ = %.1E'%(action, env.actions[action]))
                # run_trajectory(env, action = action)

            STATE = []
            CONS = []
            TCOMP = []
            TITLES2 = []
            for a_i in range(len(TITLES)):
                env.settings['Integration']['suffix'] = NAMES[a_i]
                state, cons, tcomp = load_state_files(env)
                STATE.append(state)
                CONS.append(cons)
                TCOMP.append(tcomp)
                TITLES2.append(TITLES[a_i])

            env.settings['Integration']['subfolder'] = '42_runmodel_Nvary/%iseed_%ibodies_'%(seed, Bodies[i])
            save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'Action_comparison_RL.png'
            # plot_trajs(env, STATE, CONS, TCOMP, TITLES2, save_path, plot_traj_index=traj_index)
            save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'Action_comparison_avse.png'
            plot_a_vs_e(env, STATE, CONS, TCOMP, TITLES2, save_path, plot_traj_index=traj_index)
        
    
    
    def start_env(bodies, subfolder, integrators, radius_cluster, t_step_param, \
                  hybrid = False, steps = 40):
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = subfolder +'%ibodies/'%bodies
        env.settings['Integration']['savestate'] = True
        env.settings['Integration']['max_error_accepted'] = 1e5
        env.settings['InitialConditions']['bodies_in_system'] = 'fixed'
        env.settings['Integration']['max_steps'] = steps
        env.settings['InitialConditions']['n_bodies'] = bodies

        env.settings['InitialConditions']['radius_cluster'] = radius_cluster
        env.settings['Integration']['integrator_global'] = integrators[0]
        env.settings['Integration']['integrator_local'] = integrators[1]
        env.settings['RL']['t_step_param'] = t_step_param
        env.settings['Integration']["hybrid"] = hybrid
        env._initialize_RL() # To reset actions
        return env
    
    def run_many(subfolder, integrators, radius_cluster, t_step_param = 1, hybrid = False, steps = 40):
        initializations = 10
        seeds = np.arange(initializations)
        NAMES = []
        index_to_plot = [0,2,5, 9]
        Bodies = [5, 9, 15]
        # model_path_index = ''
        model_path_index = '173'
        model_path = './Training_Results/model_weights'+model_path_index +'.pth'

        STATE_list = []
        CONS_list = []
        TCOMP_list = []
        TITLES_list = []

        for Body_i in range(len(Bodies)):
            TITLES = []

            # RL
            for i in range(initializations):
                env = start_env(Bodies[Body_i], subfolder, integrators, radius_cluster, t_step_param, hybrid = False, steps = steps)
                env.settings['InitialConditions']['seed'] = seeds[i]
                name = '_actionRL_%i'%i
                NAMES.append(name)
                env.settings['Integration']['suffix'] = name
                # run_trajectory(env, action = 'RL', model_path=model_path)
            TITLES.append(r"RL-"+model_path_index)

            # # Hybrid
            if hybrid == True:
                for i in range(initializations):
                    env = start_env(Bodies[Body_i], subfolder, integrators, radius_cluster, t_step_param, hybrid =True, steps = steps)
                    env.settings['InitialConditions']['seed'] = seeds[i]
                    name = '_actionHRL_%i'%i
                    NAMES.append(name)
                    env.settings['Integration']['suffix'] = name
                    # run_trajectory(env, action = 'RL', model_path=model_path)
                TITLES.append(r"H-RL-"+model_path_index)

            # Fixed-size
            for a_i, act in enumerate(index_to_plot): # index_toplot includes RL
                for i in range(initializations):
                    print(act, i)
                    env = start_env(Bodies[Body_i], subfolder, integrators, radius_cluster, t_step_param, hybrid = False, steps = steps)
                    env.settings['InitialConditions']['seed'] = seeds[i]
                    name = '_action_%i_%i'%(act, i)
                    NAMES.append(name)
                    env.settings['Integration']['suffix'] = name
                    # run_trajectory(env, action = act-1)
                TITLES.append(r'$\mu$ = %.1E'%(env.actions[act]*t_step_param))

            STATE = []
            CONS = []
            TCOMP = []
            for act in range(len(NAMES)):
                env.settings['Integration']['suffix'] = NAMES[act]
                state, cons, tcomp = load_state_files(env)
                STATE.append(state)
                CONS.append(cons)
                TCOMP.append(tcomp)

            env = start_env(Bodies[Body_i], subfolder, integrators, radius_cluster, t_step_param)
            save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'Energy_vs_tcomp.png'
            # plot_energy_vs_tcomp(env, STATE, CONS, TCOMP, TITLES, initializations, save_path, plot_traj_index=index_to_plot)
            # plot_energy_vs_tcomp_avg(env, STATE, CONS, TCOMP, TITLES, initializations, save_path, plot_traj_index=index_to_plot)

            STATE_list.append(STATE)
            CONS_list.append(CONS)
            TCOMP_list.append(TCOMP)
            TITLES_list.append(TITLES)
        save_path = env.settings['Integration']['savefile'] + subfolder 
        plot_energy_vs_tcomp_avg_together(env, STATE_list, CONS_list, TCOMP_list, TITLES_list, initializations, save_path, plot_traj_index=index_to_plot, hybrid = hybrid)


    if experiment == 4: # plot in tcomp vs energy together
        subfolder = '6_run_many/'
        integrators = ['Ph4', 'Huayno']
        radius_cluster = 0.1
        run_many(subfolder, integrators, radius_cluster)
    
    if experiment == 5: # plot in tcomp vs energy together long term hybrid
        subfolder = '6_run_many/'
        integrators = ['Ph4', 'Huayno']
        radius_cluster = 0.1
        run_many(subfolder, integrators, radius_cluster, hybrid = True, steps = 100)

    elif experiment == 6: # plot in tcomp vs energy together different integrators
        subfolder = '8_run_many_integrators/'
        integrators = ['Hermite', 'Ph4']
        radius_cluster = 0.1
        run_many(subfolder, integrators, radius_cluster)

    elif experiment == 7: # plot in tcomp vs energy together time-step parameter
        subfolder = '9_run_many_timestepparam/'
        integrators = ['Ph4', 'Huayno']
        radius_cluster = 0.1
        t_step_param = 1e-1
        run_many(subfolder, integrators, radius_cluster, t_step_param=t_step_param)

    # elif experiment == 8: # plot in tcomp vs energy together density
    #     subfolder = '10_run_many_density/'
    #     integrators = ['Ph4', 'Huayno']
    #     # virial_ratio = [0.1, 0.8]
    #     radius_cluster = 0.05
    #     run_many(subfolder, integrators, radius_cluster)



    
"""
TestTrainedModelGym_hermite: tests and plots for the RL algorithm

Author: Veronica Saz Ulibarrena
Last modified: 5-March-2025
"""
import numpy as np

from TrainRL import train_net
from Plots_TestTrained import plot_test_reward
from TestTrainedModel import load_reward

from env_hydro.BridgedCluster_env_hydro import Cluster_env_hydro
from TestEnvironment import run_trajectory, load_state_files
from Plots_TestEnvironment import plot_trajs_hydro


colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']

if __name__ == '__main__':
    experiment = 2 # number of the experiment to be run

    # import env
    # env = gym.make('Cluster_env_hydro-v0')

    if experiment == 0: # Train
        env = Cluster_env_hydro()
        env.settings['Integration']['subfolder'] = 'hydro_training_particles/'
        # Without pretraining
        # train_net(env = env, suffix = "currentTraining/")

        # With pretraining
        # train_net(env = env, suffix = "currentTraining/")
        model_path = env.settings['Training']['savemodel'] + 'model_weights173.pth'
        train_net(env = env, suffix = "hydro_training_particles/", model_path_pretrained = model_path)

    elif experiment == 1: # Plot training
        env = Cluster_env_hydro()
        reward, EnergyError, EnergyError_rel, HuberLoss, tcomp, testReward, trainingTime = \
                load_reward(env, suffix = 'hydro_training_particles/')
        # reward, EnergyError, EnergyError_rel, HuberLoss, tcomp, testReward, trainingTime = \
        #         load_reward(env, suffix = '24_from22_Model173_localTraining/')
        plot_test_reward(env, testReward, trainingTime)

    elif experiment == 2: # run RL and fixed for bridged nbody and hydro
        # 2: use network trained with hyperparameters chosen by hand
        env = Cluster_env_hydro()

        model_path_index = '39'
        model_path = './Training_Results/hydro_training_particles/model_weights'+model_path_index +'.pth'
        index_to_plot = [0,3, 5, 7, 9]
        env.settings['Integration']['max_steps'] = 200
        env.settings['InitialConditions']['Ndisk'] = 2000
    
        for case in range(1):
            if case == 0:
                env.settings['Integration']['hydro'] = False
                env.settings['Integration']['subfolder'] = '11_run_hydro_traj/nbody_code/Ndisk2000/'
            else:
                env.settings['Integration']['hydro'] = True
                env.settings['Integration']['subfolder'] = '11_run_hydro_traj/hydro_code/'

            NAMES = []
            TITLES = []
            NAMES.append('_actionRL')
            env.settings['Integration']['suffix'] = NAMES[0]
            TITLES.append(r'RL-39')
            run_trajectory(env, action = 'RL', model_path= model_path)
            # env.plot_orbit()
            
            for act_i, act in enumerate(index_to_plot):
                NAMES.append('_action_%0.3E'%(env.actions[act]))
                env.settings['Integration']['suffix'] = NAMES[act_i+1]
                TITLES.append(r'%i: $\Delta t_B$ = %.1E'%(act, env.actions[act]))
                run_trajectory(env, action = act)
                # env.plot_orbit()

            STATE = []
            CONS = []
            TCOMP = []
            TITLES2 = []
            for act in range(len(NAMES)):
                env.settings['Integration']['suffix'] = NAMES[act]
                state, cons, tcomp = load_state_files(env)
                STATE.append(state)
                CONS.append(cons)
                TCOMP.append(tcomp)
                TITLES2.append(TITLES[act])

            save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'Action_comparison_RL.png'
            plot_trajs_hydro(env, STATE, CONS, TCOMP, TITLES2, save_path, plot_traj_index=[0,1])

    
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

from env.BridgedCluster_env import Cluster_env
from TrainRL import train_net
from TrainingFunctions import DQN, load_reward, plot_reward
from TestEnvironment import run_trajectory, load_state_files, plot_trajs


colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']



        
if __name__ == '__main__':
    experiment = 1 # number of the experiment to be run
    seed = 1

    if experiment == 0: # Train
        train_net()

    elif experiment == 1:
        # Plot training results
        env = Cluster_env()
        reward, EnergyError, HuberLoss = load_reward(env, suffix = '')
        plot_reward(env, reward, EnergyError, HuberLoss)
        
    elif experiment == 2 or experiment == 3:
        # 2: use network trained with hyperparameters chosen by hand
        # 3: use network found by hyperparameter optimization for comparison
        env = Cluster_env()
        if experiment == 2:
            env.settings['Integration']['subfolder'] = '4_run_RL_base/'
        else:
            env.settings['Integration']['subfolder'] = '5_run_RL_hyperparam/'
            env.settings['Training']['neurons'] = 100 # TODO: fill in manually
            env.settings['Training']['layers'] = 3 # TODO: fill in manually
            env.settings['Training']['lr'] = 1e-3 # TODO: fill in manually

        NAMES = []
        NAMES.append('actionRL')
        env.settings['Integration']['suffix'] = NAMES[0]
        run_trajectory(env, action = 'RL')
        for act in range(env.settings['RL']['number_actions']):
            NAMES.append('action'+ str(env.actions[act]))
            env.settings['Integration']['suffix'] = NAMES[act+1]
            run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        for act in range(len(NAMES)):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison_RL.png'
        plot_trajs(env, STATE, CONS, TCOMP, NAMES, save_path)



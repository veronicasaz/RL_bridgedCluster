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
from Plots_TestTrained import plot_reward, plot_balance, plot_test_reward
from TestEnvironment import run_trajectory, load_state_files, plot_trajs
from Plots_TestEnvironment import plot_comparison_end, plot_distance_action, \
    plot_energy_vs_tcomp, plot_state_diff


colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']


def load_reward(a, suffix = ''):
    """
    load_reward: load rewards from file 
    INPUTS:
        a: environment
    OUTPUTS:
        score: rewards
        EnergyE: energy error
        HuberLoss: huber loss
    """
    score = []
    with open(a.settings['Training']['savemodel'] + suffix + "rewards.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            score_r = list()
            for j in y.split():
                score_r.append(float(j))
            score.append(score_r)

    EnergyE = []
    with open(a.settings['Training']['savemodel'] + suffix + "EnergyError.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            Energy_r = list()
            for j in y.split():
                Energy_r.append(float(j))
            EnergyE.append(Energy_r)

    EnergyE_rel = []
    with open(a.settings['Training']['savemodel'] + suffix + "EnergyError_rel.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            Energy_rel_r = list()
            for j in y.split():
                Energy_rel_r.append(float(j))
            EnergyE_rel.append(Energy_rel_r)


    tcomp = []
    with open(a.settings['Training']['savemodel'] + suffix + "Tcomp.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            tcomp_r = list()
            for j in y.split():
                tcomp_r.append(float(j))
            tcomp.append(tcomp_r)

    testReward = []
    with open(a.settings['Training']['savemodel'] + suffix + "TestReward.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            testreward_r = list()
            for j in y.split():
                testreward_r.append(float(j))
            testReward.append(testreward_r)

    HuberLoss = []

    return score, EnergyE, EnergyE_rel, HuberLoss, tcomp, testReward

if __name__ == '__main__':
    experiment = 0 # number of the experiment to be run
    seed = 1

    if experiment == 0: # Train
        env = Cluster_env()
        env.settings['Training']['RemovePlanets'] = True # train without planets (they do not contribute to the total energy error)
        train_net(suffix = "currentTraining/")

    elif experiment == 1:
        # Plot training results
        env = Cluster_env()
        reward, EnergyError, EnergyError_rel, HuberLoss, tcomp, testReward = load_reward(env, suffix = 'currentTraining/')
        # plot_reward(env, reward, EnergyError, HuberLoss)
        # plot_balance(env, reward, EnergyError, EnergyError_bridge, tcomp)
        plot_test_reward(env, testReward)
        
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

        model_path_index = '79'
        model_path = './Training_Results/model_weights'+model_path_index +'.pth'
        index_to_plot = [0, 1,3,5,  8, 10]

        env.settings['Training']['RemovePlanets'] = True
        env.settings['Integration']['max_error_accepted'] = 1e5
        
        NAMES = []
        NAMES.append('_actionRL')
        env.settings['Integration']['suffix'] = NAMES[0]
        # run_trajectory(env, action = 'RL', model_path= model_path)
        for act in range(env.settings['RL']['number_actions']):
        # for act in index_to_plot:
            NAMES.append('_action'+ str(env.actions[act]))
            env.settings['Integration']['suffix'] = NAMES[act+1]
            # run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        TITLES = []
        # for act in range(len(NAMES)):
        for act in index_to_plot:
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)
            TITLES.append(NAMES[act])

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison_RL.png'
        plot_trajs(env, STATE, CONS, TCOMP, TITLES, save_path, plot_traj_index=[0,1])
        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison_RL2.png'
        # plot_distance_action(env, STATE, CONS, TCOMP, NAMES, save_path)
        plot_state_diff(env, STATE, CONS, TCOMP, TITLES, save_path)
        # save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            # 'Action_comparison_RL3.png'
        # plot_comparison_end(env, STATE, CONS, TCOMP, NAMES, save_path, plot_traj_index=[0,1])

    elif experiment == 4:
        def start_env():
            env = Cluster_env()
            env.settings['Integration']['subfolder'] = '6_run_many/'
            env.settings['Integration']['max_steps'] = 60
            env.settings['Integration']['savestate'] = True
            return env

        initializations = 10
        seeds = np.arange(initializations)
        NAMES = []


        for i in range(initializations):
            env = start_env()
            env.settings['InitialConditions']['seed'] = seeds[i]
            NAMES.append('_actionRL_%i'%i)
            print(NAMES)
            env.settings['Integration']['suffix'] = NAMES[i]
            # run_trajectory(env, action = 'RL')

        for act in range(env.settings['RL']['number_actions']):
            for i in range(initializations):
                print(act, i)
                env = start_env()
                env.settings['InitialConditions']['seed'] = seeds[i]
                name = '_action_%i_%i'%(act, i)
                NAMES.append(name)
                env.settings['Integration']['suffix'] = name
                # run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        for act in range(len(NAMES)):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        env = start_env()
        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Energy_vs_tcomp.png'
        plot_energy_vs_tcomp(env, STATE, CONS, TCOMP, NAMES, initializations, save_path, plot_traj_index=[0, 1, 2, 3, 4])



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
from TrainRL import DQN
from Plots_TestEnvironment import plot_trajs, plot_intializations, plot_rewards, plot_reward_comparison,\
    plot_convergence, plot_trajs_reward


colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']

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
    
    elif action == 'random':
        while terminated == False:
            action_i = np.random.randint(len(env.actions), size = 1)
            x, y, terminated, zz = env.step(action_i)
            i += 1
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


if __name__ == '__main__':
    experiment = 2 # number of the experiment to be run
            
    if experiment == 0: #test creation of planetary systems
        
        seed = np.random.randint(10000, size = 1000)
        # seed = [6496]
        for i in range(len(seed)):
            print(i, seed[i])
            env = Cluster_env()
            env.settings['Integration']['savestate'] = False
            env.settings['InitialConditions']['seed'] = seed[i]
            state, info = env.reset()
            env.close()
    
    elif experiment == 1: # run bridge for all actions
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '1_run_actions/'
        env.settings['InitialConditions']['seed'] = 1
        env.settings['Training']['RemovePlanets'] = False

        NAMES = []
        # for act in range(1):
        for act in range(env.settings['RL']['number_actions']):
            print("Action", env.actions[act])
            name = '_action'+str(act)
            NAMES.append(name)
            env.settings['Integration']['suffix'] = NAMES[act]
            # run_trajectory(env, action = act)

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

    elif experiment == 2: # run convergence study
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '1_run_convergence/seed1_2/'
        env.settings['InitialConditions']['seed'] = 1
        env.settings['Training']['RemovePlanets'] = False
        env.settings['Integration']['max_steps'] = 100
        env.settings['Integration']["max_error_accepted"] = 1e10


        max_actions = 9
        env.settings['RL']['number_actions'] = 4 #limit how many actions we choose
        env.settings['RL']["range_action"] = [1e-4, 1e-2]
        env._initialize_RL()

        actions = np.zeros(max_actions)
        prev_action = env.actions[0]
        for act in reversed(range(5)):
            actions[act] = prev_action/2
            prev_action = actions[act]
        actions[5:] = env.actions
        env.actions = actions
        env.settings['RL']['number_actions'] = max_actions

        NAMES = []
        for act in reversed(range(env.settings['RL']['number_actions'])):
            print("Action", env.actions[act])
            name = '_action_%0.2E'%(env.actions[act])
            NAMES.append(name)
            env.settings['Integration']['suffix'] = name
            run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        TITLES = []
        for act in range(env.settings['RL']['number_actions']):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)
            TITLES.append(r"$\Delta t$ = %.2E Myr"%(env.actions[act]))

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison'
        plot_convergence(env, STATE, CONS, TCOMP, TITLES, save_path)
        plot_trajs_reward(env, STATE, CONS, TCOMP, TITLES, save_path)
    
    elif experiment == 3: # put together convergence study for different seeds
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '1_run_convergence/seed1_2/'
        env.settings['InitialConditions']['seed'] = 1
        env.settings['Training']['RemovePlanets'] = False
        env.settings['Integration']['max_steps'] = 100
        env.settings['Integration']["max_error_accepted"] = 1e10


        max_actions = 9
        env.settings['RL']['number_actions'] = 4 #limit how many actions we choose
        env.settings['RL']["range_action"] = [1e-4, 1e-2]
        env._initialize_RL()

        actions = np.zeros(max_actions)
        prev_action = env.actions[0]
        for act in reversed(range(5)):
            actions[act] = prev_action/2
            prev_action = actions[act]
        actions[5:] = env.actions
        env.actions = actions
        env.settings['RL']['number_actions'] = max_actions

        NAMES = []
        for act in reversed(range(env.settings['RL']['number_actions'])):
            print("Action", env.actions[act])
            name = '_action_%0.2E'%(env.actions[act])
            NAMES.append(name)
            env.settings['Integration']['suffix'] = name
            run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        TITLES = []
        for act in range(env.settings['RL']['number_actions']):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)
            TITLES.append(r"$\Delta t$ = %.2E Myr"%(env.actions[act]))

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison'
        plot_convergence(env, STATE, CONS, TCOMP, TITLES, save_path)
        plot_trajs_reward(env, STATE, CONS, TCOMP, TITLES, save_path)

    elif experiment == 4: # test different initializations
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '2_run_initializations/'

        NAMES = []
        seeds = [1, 2, 3, 4]
        action = 5 # action = 1e-4

        for i in range(len(seeds)):
            name = '_seed'+ str(seeds[i])+'_action'+str(action)
            NAMES.append(name)
            env.settings['Integration']['suffix'] = name
            env.settings['Integration']['max_steps'] = 50
            env.settings['InitialConditions']['seed'] = seeds[i]
            env.settings['Training']["RemovePlanets"]= False
            env.settings['Integration']["max_error_accepted"] = 1e10
            # run_trajectory(env, action = action) 

        STATE = []
        CONS = []
        TCOMP = []
        TITLES = []
        for i in range(len(NAMES)):
            env.settings['Integration']['suffix'] = NAMES[i]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)
            TITLES.append(r"Seed %i"%(seeds[i]))

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Initialization_comparison.png'
        plot_intializations(env, STATE, CONS, TCOMP, TITLES, save_path)
        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Reward_comparison.png'
        # plot_rewards(env, STATE, CONS, TCOMP, NAMES, save_path)

    elif experiment == 5: # test rewards
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
        plot_reward_comparison(env, [STATE[action]], [CONS[action]], [TCOMP[action]], [NAMES[action]], save_path)






        
        
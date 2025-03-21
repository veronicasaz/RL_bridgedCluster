"""
TestEnvironment: tests simulation environments

Author: Veronica Saz Ulibarrena
Last modified: 5-March-2025
"""
import numpy as np
import torch

from env.BridgedCluster_env import Cluster_env
from TrainRL import DQN
from Plots_TestEnvironment import plot_trajs, plot_trajs_noRL,\
    plot_intializations, plot_rewards, plot_reward_comparison,\
    plot_convergence, plot_trajs_reward, plot_convergence_togetherseeds,\
    plot_convergence_direct_integra


colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']

def run_trajectory(env, action = 'RL', model_path = None, bridge = True):
    """
    run_trajectory: Run one initialization with RL or with an integrator
    INPUTS:
        env: environment to simulate
        action: fixed action or 'RL' 
        model_path: path to the trained RL algorithm
        bridge: True or False 
    """
    if model_path == None:
        model_path = env.settings['Training']['savemodel'] +'model_weights.pth'
        
    if bridge == True:
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
            try:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = model(state).max(1)[1].view(1, 1)
                state, y, terminated, info = env.step(action.item())
            except:
                env.plot_orbit()
        env.close()
    
    elif action == 'random':
        while terminated == False:
            action_i = np.random.randint(len(env.actions), size = 1)
            x, y, terminated, zz = env.step(action_i)
            i += 1
        env.close()
    
    # Case 3: fixed action throughout the simulation
    if bridge == False:
        env.reset_withoutBridge()
        while terminated == False:
            x, y, terminated, zz = env.step_withoutBridge(action)
            i += 1
        env.close_withoutBridge()
    else:
        while terminated == False:
            try:
                x, y, terminated, zz = env.step(action)
            except:
                env.plot_orbit()
            i += 1
        env.close()

def load_state_files(env, namefile = None):
    """
    load_state_files: Load run information 
    INPUTS: 
        env: environment of the saved files
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
    experiment = 7 # number of the experiment to be run
            
    if experiment == 0: #test creation of planetary systems
        seed = np.random.randint(10000, size = 1000)
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
        env.settings['InitialConditions']['seed'] = 3
        env.settings['Training']['RemovePlanets'] = False
        env.settings['InitialConditions']['bodies_in_system'] = "fixed"
        env.settings['InitialConditions']['nbodies'] = 9
        env.settings['Integration']['max_steps'] = 40
        env.settings['Integration']["max_error_accepted"] = 1e10
        env.settings['Integration']["hybrid"] = False

        NAMES = []
        for act in range(1):
            print("Action", env.actions[act+4])
            name = '_action'+str(act) +str(env.settings['Integration']["hybrid"])
            NAMES.append(name)
            env.settings['Integration']['suffix'] = name
            run_trajectory(env, action = act+2)

        env.settings['Integration']["hybrid"] = True
            
        for act in range(1):
            print("Action", env.actions[act])
            name = '_action'+str(act) +str(env.settings['Integration']["hybrid"])
            NAMES.append(name)
            env.settings['Integration']['suffix'] = name
            run_trajectory(env, action = act+2)

        STATE = []
        CONS = []
        TCOMP = []
        TITLES = ['No hybrid', "hybrid"]
        for act in range(len(NAMES)):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison.png'
        plot_trajs(env, STATE, CONS, TCOMP, TITLES, save_path)

    elif experiment == 2: # run convergence study
        for seed in [1, 2, 3, 4]:
            env = Cluster_env()
            env.settings['Integration']['subfolder'] = '1_run_convergence/seed%i/'%seed
            env.settings['InitialConditions']['seed'] = seed
            env.settings['Training']['RemovePlanets'] = False
            env.settings['Integration']['max_steps'] = 40
            env.settings['Integration']["max_error_accepted"] = 1e10
            env.settings['Integration']["bridge"] = 'modified'

            max_actions = 9
            env.settings['RL']['number_actions'] = 5 #limit how many actions we choose
            env.settings['RL']["range_action"] = [1e-6, 1e-2]
            env._initialize_RL()

            NAMES = []
            for act in range(env.settings['RL']['number_actions']):
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
        env.settings['Integration']['subfolder'] = '1_run_convergence/'

        max_actions = 9
        env.settings['RL']['number_actions'] = 5 #limit how many actions we choose
        env.settings['RL']["range_action"] = [1e-6, 1e-2]
        env._initialize_RL()

        NAMES = []
        for act in range(env.settings['RL']['number_actions']):
            name = '_action_%0.2E'%(env.actions[act])
            NAMES.append(name)

        seed_folder = ['seed1/', 'seed2/', 'seed3/', 'seed4/']
        STATE_list = []
        CONS_list = []
        TCOMP_list = []
        TITLES_list = []
        for j in range(len(seed_folder)):
            env.settings['Integration']['subfolder'] = '1_run_convergence/' + seed_folder[j]
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

            STATE_list.append(STATE)
            CONS_list.append(CONS)
            TCOMP_list.append(TCOMP)
            TITLES_list.append(TITLES)

        save_path = env.settings['Integration']['savefile'] + '1_run_convergence/'+\
            'Convergence_comparison'
        plot_convergence_togetherseeds(env, STATE_list, CONS_list, TCOMP_list, TITLES_list, save_path)
        
    elif experiment == 4: # test different initializations
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '2_run_initializations/'

        NAMES = []
        seeds = np.arange(4)+1
        action = 0 # action = 1e-4

        for i in range(len(seeds)):
            name = '_seed'+ str(seeds[i])+'_action'+str(action)
            NAMES.append(name)
            env.settings['Integration']['suffix'] = name
            env.settings['Integration']['max_steps'] = 40
            env.settings['InitialConditions']['seed'] = seeds[i]
            env.settings['Training']["RemovePlanets"]= False
            env.settings['Integration']["max_error_accepted"] = 1e10
            # run_trajectory(env, action = action, bridge = True) 

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

    elif experiment == 6: # create baseline without bridge
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '1_run_actions_woBridge/'
        env.settings['InitialConditions']['seed'] = 1
        env.settings['Training']['RemovePlanets'] = False
        env.settings['Integration']['max_steps'] = 40
        env.settings['Integration']["max_error_accepted"] = 1e10
        env.settings['InitialConditions']["bodies_in_system"] = 'fixed'
        env.settings['InitialConditions']['n_bodies'] = 10

        NAMES = []
        TITLES = []

        action = 2
        # Without bridge
        name = '_nobridge'
        NAMES.append(name) 
        TITLES.append("Direct code")
        env.settings['Integration']['suffix'] = NAMES[0]
        run_trajectory(env, action = action, bridge = False) # Action does not affect

        # # With bridge
        name = '_orbridge'
        TITLES.append("Bridge")
        NAMES.append(name)
        env.settings['Integration']['suffix'] = NAMES[1]
        env.settings['Integration']["bridge"] = 'original'
        run_trajectory(env, action = action, bridge = True) # Action does not affect

        # With modified bridge
        name = '_modbridge'
        TITLES.append("Inclusive Bridge")
        NAMES.append(name)
        env.settings['Integration']['suffix'] = NAMES[2]
        env.settings['Integration']["bridge"] = 'modified'
        run_trajectory(env, action = action, bridge = True) # Action does not affect

        STATE = []
        CONS = []
        TCOMP = []
        for act in range(3):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison.png'
        plot_trajs_noRL(env, STATE, CONS, TCOMP, TITLES, save_path, subplots=3)

    elif experiment == 7: # Test direct numerical method for different number bodies
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '1_run_directintegration/'
        env.settings['Integration']['max_steps'] = 40
        env.settings['Integration']["max_error_accepted"] = 1e10

        NAMES = []
        TITLES = []

        bodies_list = [5, 10, 50, 100, 200]
        seeds = [3]

        for SED in seeds:
            env.settings['InitialConditions']['seed'] = SED
            env.settings['InitialConditions']["bodies_in_system"] = "Fixed"

            # ph4
            name = '_nobridge_'
            for bodies in bodies_list:
                env.settings['InitialConditions']["n_bodies"] = bodies
                namei = str(SED) + name + str(bodies)
                NAMES.append(namei) 
                TITLES.append('Direct %i'%bodies)
                env.settings['Integration']['suffix'] = namei
                # run_trajectory(env, action = 0, bridge = False) # Action does not affect

            name = '_modbridge_fast_'
            env.settings['Integration']["hybrid"] = False
            for bodies in bodies_list:
                env.settings['InitialConditions']["n_bodies"] = bodies
                namei = str(SED) + name + str(bodies)
                NAMES.append(namei) 
                TITLES.append('Bridge %i'%bodies)
                env.settings['Integration']['suffix'] = namei
                # run_trajectory(env, action = 4, bridge = True) 

            
            name = '_modbridge_accurate_'
            env.settings['Integration']["hybrid"] = True
            for bodies in bodies_list:
                env.settings['InitialConditions']["n_bodies"] = bodies
                namei = str(SED) + name + str(bodies)
                NAMES.append(namei) 
                TITLES.append('H-Bridge %i'%bodies)
                env.settings['Integration']['suffix'] = namei
                # run_trajectory(env, action = 4, bridge = True) 

            name = '_modbridgeRL_'
            env.settings['Integration']["hybrid"] = False
            model_path_index = '173'
            model_path = './Training_Results/model_weights'+model_path_index +'.pth'
            for bodies in bodies_list:
                env.settings['InitialConditions']["n_bodies"] = bodies
                namei = str(SED) + name + str(bodies)
                NAMES.append(namei) 
                TITLES.append('RL Bridge %i'%bodies)
                env.settings['Integration']['suffix'] = namei
                # run_trajectory(env, action = 'RL', bridge = True, model_path = model_path) # Action does not affect

            name = '_modbridgeHRL_'
            env.settings['Integration']["hybrid"] = True
            model_path = './Training_Results/model_weights'+model_path_index +'.pth'
            for bodies in bodies_list:
                env.settings['InitialConditions']["n_bodies"] = bodies
                namei = str(SED) + name + str(bodies)
                NAMES.append(namei) 
                TITLES.append('H-RL Bridge %i'%bodies)
                env.settings['Integration']['suffix'] = namei
                # run_trajectory(env, action = 'RL', bridge = True, model_path = model_path) # Action does not affect

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
            'Bodies_comparison.png'
        plot_convergence_direct_integra(env, bodies_list, seeds,STATE, CONS, TCOMP, TITLES, save_path)
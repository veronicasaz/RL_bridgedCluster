import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import clear_output
import gym
import torch
torch.manual_seed(0)
import torch.optim as optim
from collections import namedtuple, deque
from itertools import count
import pathlib
import csv

from optimization_algorithms import EvolAlgorithm
from helpfunctions import load_json
from TrainingFunctions import DQN, \
                            ReplayMemory,\
                            select_action, \
                            optimize_model
from TestTrainedModel import plot_trajs
from  ENVS.bridgedparticles.envs.Bridged3Body_env import ThreeBody_env
from TestEnvironment import load_state_files, run_trajectory



def train_sample(x, f_args):
    env, EA_iter, ind = f_args
    max_iter = int(x[0])
    layers = int(x[1])
    neurons = int(x[2])
    LR = x[3]
    print("Parameters", x)

    # env = gym.make('bridgedparticles:ThreeBody-v0') # create the env once it's been registered

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    # TRAINING settings
    BATCH_SIZE = env.settings['Training']['batch_size'] # number of transitions sampled from the replay buffer
    GAMMA = env.settings['Training']['gamma'] # discount factor
    EPS_START = env.settings['Training']['eps_start'] # starting value of epsilon
    EPS_END = env.settings['Training']['eps_end'] # final value of epsilon
    EPS_DECAY = env.settings['Training']['eps_decay'] # controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = env.settings['Training']['tau'] # update rate of the target network
    env.settings['Integration']['savestate'] = False
    # LR = settings['Training']['lr'] # learning rate of the ``AdamW`` optimizer

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    n_observations = env.observation_space.n # TODO: test

    # Get the number of state observations
    n_observations = len(state)

    # Create nets
    policy_net = DQN(n_observations, n_actions, neurons, layers).to(device)
    target_net = DQN(n_observations, n_actions, neurons, layers).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward'))

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000, Transition = Transition)

    state, info = env.reset()
    env.settings['Training']['savemodel'] = "./Training_Results/hyperparam_optim/"
    env.settings['Training']['display'] = False

    episode_number = 0 # counter of the number of steps

    # lists to save training progress
    save_reward = list()
    save_EnergyE = list()
    save_huberloss = list()

    # Training loop
    while episode_number <= max_iter:
        print("EA episode: %i, individual: %i,  Training episode: %i/%i"%(EA_iter, ind,  episode_number, max_iter))

        # Initialize the environment and get it's state
        state, info = env.reset()

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        save_reward_list = list()
        save_EnergyE_list = list()
        save_huberloss_list = list()

        # Do first step without updating the networks and with the best step
        action, steps_done = select_action(state, policy_net, [EPS_START, EPS_END, EPS_DECAY], env, device, steps_done)
        observation, reward_p, terminated, info = env.step(action.item())

        terminated = False
        while terminated == False:
            # Take a step
            action, steps_done = select_action(state, policy_net, [EPS_START, EPS_END, EPS_DECAY], env, device, steps_done)
            observation, reward_p, terminated, info = env.step(action.item())
            save_reward_list.append(reward_p)
            save_EnergyE_list.append(info['Energy_error'])

            reward = torch.tensor([reward_p], device=device)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            Huber_loss = optimize_model(policy_net, target_net, memory, \
                    Transition, device, GAMMA, BATCH_SIZE,\
                    optimizer)
            
            if Huber_loss == None:
                save_huberloss_list.append(0)
            else:
                save_huberloss_list.append(Huber_loss.item())

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if terminated:
                break

        episode_number += 1

        save_reward.append(save_reward_list)
        save_EnergyE.append(save_EnergyE_list)
        save_huberloss.append(save_huberloss_list)
        
        # if i_episode > 0 and (i_episode %100 == 0 or i_episode == (max_iter - 1)):
        if episode_number == (max_iter - 1):
            torch.save(policy_net.state_dict(), env.settings['Training']['savemodel'] +str(EA_iter) +'_'+str(ind)+'_model_weights'+str(episode_number)+'.pth') # save model

            with open(env.settings['Training']['savemodel']+"rewards.txt", "w") as f:
                for ss in save_reward:
                    for s in ss:
                        f.write(str(s) +" ")
                    f.write("\n")

            with open(env.settings['Training']['savemodel']+"EnergyError.txt", "w") as f:
                for ss in save_EnergyE:
                    for s in ss:
                        f.write(str(s) +" ")
                    f.write("\n")

            with open(env.settings['Training']['savemodel']+"HuberLoss.txt", "w") as f:
                for ss in save_huberloss:
                    for s in ss:
                        f.write(str(s) +" ")
                    f.write("\n")
        env.close()
        
    return -save_reward[-1][-1] # maximize reward for rl, minimize for EA
    
def plot_results(path, ind, m_iter):
    pop = np.genfromtxt(path, delimiter=',')
    index_min = np.where(pop[:,2] == min(pop[:,2]))[0]
    # for i in range(m_iter):
    for i in range(np.shape(pop)[0]//ind):
        plt.scatter(np.ones(ind)*i, pop[i*ind:i*ind+ind, 2])
    plt.ylabel("Fitness: test MSE ")
    plt.xlabel("Iteration")
    # plt.yscale("symlog")
    plt.savefig(path + "fitness.png", dpi = 100)
    plt.show()

def find_file(text, names):
    # print(text)
    # print(names)
    options = [i for i in names if i.startswith(text)]
    if len(options):
        return options
    else:
        return None
    
def test_results(path, ind, m_iter):
    steps = 80
    seed = 1
    n_plots = 8
    
    env = ThreeBody_env()
    env.subfolder = '2_TrainingResults/'

    # Load files without RL
    state_nobridge_Hermite, cons_nobridge_Hermite, tcomp_nobridge_Hermite = load_state_files(env, steps, namefile = '_nobridge_Hermite')
    state_nobridge_Huayno, cons_nobridge_Huayno, tcomp_nobridge_Huayno = load_state_files(env, steps, namefile = '_nobridge_Huayno')
    STATES_nobridge = [state_nobridge_Hermite, state_nobridge_Huayno]
    TITLES_nobridge = ['No bridge Hermite', 'No bridge Huayno']
    PATHS_nobridge = ['_nobridge_Hermite', '_nobridge_Huayno']

    state_bridge_00, cons_bridge_00, tcomp_bridge_00 = load_state_files(env, steps, namefile = '_bridge_b0_i0')
    state_bridge_30, cons_bridge_30, tcomp_bridge_30 = load_state_files(env, steps, namefile = '_bridge_b3_i0')
    state_bridge_22, cons_bridge_22, tcomp_bridge_22 = load_state_files(env, steps, namefile = '_bridge_b2_i2')
    state_bridge_03, cons_bridge_03, tcomp_bridge_03 = load_state_files(env, steps, namefile = '_bridge_b0_i3')
    state_bridge_33, cons_bridge_33, tcomp_bridge_33 = load_state_files(env, steps, namefile = '_bridge_b3_i3')
    
    STATES_bridge = [state_bridge_00, state_bridge_30, state_bridge_22,state_bridge_03, state_bridge_33]
    CONS_bridge = [cons_bridge_00, cons_bridge_30, cons_bridge_22,cons_bridge_03, cons_bridge_33]
    TCOMP_bridge = [tcomp_bridge_00, tcomp_bridge_30, tcomp_bridge_22,tcomp_bridge_03, tcomp_bridge_33]
    TITLES_bridge =  [ r'Bridge 00', r'Bridge 30', r'Bridge 22', r'Bridge 03', r'Bridge 33']
    PATHS_bridge = ['_bridge_b0_i0','_bridge_b3_i0','_bridge_b2_i2','_bridge_b0_i3','_bridge_b3_i3']

    ######################################
    # Create files with RL
    env = ThreeBody_env()
    path_save = env.settings["Training"]['savemodel'] + 'hyperparam_optim/run/'

    file_names = [str(f) for f in pathlib.Path(env.settings['Training']['savemodel'] +\
                    'hyperparam_optim/').iterdir() if f.is_file()] 
    
    path_csv = env.settings["Training"]['savemodel'] + 'hyperparam_optim/evol_population.csv'
    PARAMS = []
    with open(path_csv, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            PARAMS.append([ float(i) for i in row ])
    PARAMS = np.array(PARAMS)

    # Choose 8 best networks
    all_indexes = np.copy(PARAMS[:, 2])
    indexes = np.zeros(n_plots)
    for i in range(len(indexes)):
        indexes[i] = int(np.where(all_indexes == min(all_indexes))[0])
        all_indexes[int(indexes[i])] = max(all_indexes) + 1 # large value to avoid it being the minimum

    indexes = [int(x) for x in indexes]
    PARAMS_2 = PARAMS[indexes, :]

    for j in range(n_plots):
        # env = ThreeBody_env()
        # env.subfolder = '2_TrainingResults/'
        env.bridged = True
        env.integrator = 'Hermite'
        env.save_path = path_save
            
        # layers = int(float(PARAMS[ind*(m_iter+1)+j][2]))
        # neurons = int(float(PARAMS[ind*(m_iter+1) +j][3]))
        layers = int(float(PARAMS_2[j][4]))
        neurons = int(float(PARAMS_2[j][5]))

        # Load trained model
        # text = env.settings['Training']['savemodel'][2:] +\
        #             'hyperparam_optim/'+ str(m_iter) + '_'+str(j)+'_'+\
        #             'model_weights'
        print(j, indexes)
        text = env.settings['Training']['savemodel'][2:] +\
                    'hyperparam_optim/'+ str(int(PARAMS_2[j, 0])) + '_'+str(int(PARAMS_2[j, 1]))+'_'+\
                    'model_weights'
        print(text)
        model_path = './' + find_file( text, file_names)[0] 
        run_trajectory(seed = seed, action = 'RL', env = env,\
                            name_suffix = '_bridge_RL_' + str(j), steps = steps,
                            model_path = model_path, architecture = [layers, neurons],
                            save_path = path_save, steps_suffix = str(j))
        

    # Load files with RL and plot each of them
    STATES_RL = []
    CONS_RL = []
    TCOMP_RL = []
    TITLES_RL = []
    PATHS_RL = []
    for j in range(n_plots):
        state_bridge_RL, cons_bridge_RL, tcomp_bridge_RL = load_state_files(env, steps, namefile = '_bridge_RL_' + str(j))
        title_RL = 'RL ' + str(j)
        path_RL = '_bridge_RL_' + str(j)
        STATES_RL.append(state_bridge_RL)
        CONS_RL.append(cons_bridge_RL)
        TCOMP_RL.append(tcomp_bridge_RL)
        TITLES_RL.append(title_RL)
        PATHS_RL.append(path_RL)

        plot_save = env.settings["Training"]['savemodel'] + 'hyperparam_optim/run/' + str(j) + '_'
        steps_taken = np.load(path_save + 'RL_steps_taken' +str(j)+'.npy', allow_pickle=True)
        
        plot_trajs(env,\
                    STATES_bridge+ [state_bridge_RL],
                    CONS_bridge+ [cons_bridge_RL], \
                    TCOMP_bridge+ [tcomp_bridge_RL], \
                    TITLES_bridge + [title_RL],\
                    PATHS_bridge + [path_RL], \
                    steps_taken = steps_taken,\
                    save_path = plot_save, 
                    steps = steps)

if __name__ == '__main__':
    settings_file_path= "./ANN_settings.json"
    settings = load_json("./settings_integration_3Body.json")
    path_model = "./Training_Results/hyperparam_optim/"

    # Evolutionary Algorithm
    ind = settings['Hyperparam_optim']['individuals']
    m_iter = settings['Hyperparam_optim']['max_iter']
    bnds = settings['Hyperparam_optim']['bnds']

    bnds_list = list()
    for key, val in bnds.items():
        bnds_list.append(val)
    bnds_list = np.array(bnds_list)
    
    typex = ['int', 'int', 'int', 'log']
    f_args = [path_model, settings]

    x_minVal, lastMin = EvolAlgorithm(train_sample, bnds_list, x_add = f_args, \
                    typex = typex, bulk_fitness = False,\
                    ind = ind, max_iter = m_iter, 
                    path_save = path_model)
    print("X min", x_minVal)
    print('Min', lastMin)

    plot_results("./Training_Results/hyperparam_optim/evol_population.csv", ind, m_iter)
    test_results("./Training_Results/hyperparam_optim/", ind, m_iter)
"""
TestGym: Training of the reinforcement learning algorithm

Author: Veronica Saz Ulibarrena
Last modified: 6-February-2024

Based on:
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/https://www.gymlibrary.dev/content/environment_creation/
https://www.gymlibrary.dev/content/environment_creation/
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
"""
from IPython.display import clear_output

import gym
import torch

import matplotlib.pyplot as plt
import matplotlib
import torch.optim as optim

from collections import namedtuple, deque
from itertools import count

from helpfunctions import load_json
from TrainingFunctions import DQN, \
                            ReplayMemory,\
                            select_action, \
                            optimize_model,\
                            plot_durations


def train_net(env = None, suffix = ''):
    # Environment
    if env == None:
        # env = gym.make('bridgedparticles:ThreeBody-v0') # create the env once it's been registered
        env = gym.make('bridgedparticles:BS-v0') # create the env once it's been registered

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    # settings = load_json("./settings_symple.json")
    
    # TRAINING settings
    BATCH_SIZE = env.settings['Training']['batch_size'] # number of transitions sampled from the replay buffer
    GAMMA = env.settings['Training']['gamma'] # discount factor
    EPS_START = env.settings['Training']['eps_start'] # starting value of epsilon
    EPS_END = env.settings['Training']['eps_end'] # final value of epsilon
    EPS_DECAY = env.settings['Training']['eps_decay'] # controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = env.settings['Training']['tau'] # update rate of the target network
    LR = env.settings['Training']['lr'] # learning rate of the ``AdamW`` optimizer
    NEURONS = env.settings['Training']['neurons']
    LAYERS = env.settings['Training']['hidden_layers']
    env.settings['Integration']['savestate'] = False

    # Get number of actions from gym action space
    n_actions = env.action_space.n # TODO: test

    # Get the number of state observations
    # n_observations = len(env.observation_space) # TODO: test
    n_observations = env.observation_space.n # TODO: test

    # Create nets
    policy_net = DQN(n_observations, n_actions, NEURONS, LAYERS).to(device)
    target_net = DQN(n_observations, n_actions, NEURONS, LAYERS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward'))

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000, Transition = Transition)

    
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    state, info = env.reset()
    episode_number = 0 # counter of the number of steps

    # lists to save training progress
    save_reward = list()
    save_EnergyE = list()
    save_huberloss = list()

    # Training loop
    while episode_number <= env.settings['Training']['max_episodes']:
        print("Training episode: %i/"%episode_number)

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
        
        if episode_number %100 == 0:
            torch.save(policy_net.state_dict(), env.settings['Training']['savemodel'] +suffix+ 'model_weights'+str(episode_number)+'.pth') # save model
        else:
            torch.save(policy_net.state_dict(), env.settings['Training']['savemodel'] +suffix+ 'model_weights.pth') # save model

        # save training
        with open(env.settings['Training']['savemodel']+suffix+"rewards.txt", "w") as f:
            for ss in save_reward:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"EnergyError.txt", "w") as f:
            for ss in save_EnergyE:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"HuberLoss.txt", "w") as f:
            for ss in save_huberloss:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

    env.close()
    print('Complete')

if __name__ == '__main__':
    train_net()